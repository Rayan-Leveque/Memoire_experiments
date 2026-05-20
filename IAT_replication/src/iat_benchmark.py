#!/usr/bin/env python3
"""
iat_benchmark.py
Main orchestrator for LLM IAT bias evaluation
Loads models sequentially, runs all IAT trials, saves results per model
"""

import os
import pandas as pd
import random
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from vllm import LLM, SamplingParams

from config_loader import ConfigLoader, ModelConfig


@dataclass
class IATStimuli:
    """Represents IAT stimuli for a category/dataset"""
    category: str
    dataset: str
    concept_a_list: List[str]
    concept_b_list: List[str]
    attributes: List[str]


class IATStimuliLoader:
    """Loads and parses IAT stimuli CSV with A/B inheritance logic"""
    
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.stimuli = self._parse_stimuli()
    
    def _parse_stimuli(self) -> Dict[str, IATStimuli]:
        """Parse CSV with A/B inheritance logic"""
        stimuli = {}
        
        for category in self.df['category'].unique():
            cat_df = self.df[self.df['category'] == category]
            
            for dataset in cat_df['dataset'].unique():
                dataset_df = cat_df[cat_df['dataset'] == dataset]
                
                # Get concept pairs (A, B) - filter out empty values
                concept_a = dataset_df['A'].dropna().unique().tolist()
                concept_b = dataset_df['B'].dropna().unique().tolist()
                
                if len(concept_a) == 0 or len(concept_b) == 0:
                    print(f"⚠️  Warning: No concept pairs for {category}/{dataset}, skipping")
                    continue
                
                # Get all attributes (C) - lowercase and filter empty
                attributes = dataset_df['C'].dropna().str.strip().str.lower().tolist()
                attributes = [a for a in attributes if a]  # Remove empty strings
                
                if len(attributes) == 0:
                    print(f"⚠️  Warning: No attributes for {category}/{dataset}, skipping")
                    continue
                
                key = f"{category}_{dataset}"
                stimuli[key] = IATStimuli(
                    category=category,
                    dataset=dataset,
                    concept_a_list=concept_a,
                    concept_b_list=concept_b,
                    attributes=attributes
                )
        
        return stimuli
    
    def get_stimuli_by_category(self, category: str) -> List[IATStimuli]:
        """Get all stimuli for a specific category"""
        return [s for s in self.stimuli.values() if s.category == category]
    
    def get_all_stimuli(self) -> List[IATStimuli]:
        """Get all stimuli"""
        return list(self.stimuli.values())


class IATBenchmark:
    """Main IAT benchmark orchestrator"""
    
    def __init__(self, config_path: str = "config/models_config.yml"):
        print("="*80)
        print("IAT Benchmark - LLM Implicit Bias Evaluation")
        print("="*80)
        
        # Load configuration
        self.config = ConfigLoader(config_path)
        self.eval_config = self.config.get_evaluation_config()
        
        # Load stimuli
        print(f"\n📂 Loading stimuli from {self.eval_config['stimuli_path']}...")
        self.stimuli_loader = IATStimuliLoader(self.eval_config['stimuli_path'])
        
        # Load prompt templates
        self.prompt_templates = self.config.get_prompt_templates()
        print(f"✅ Loaded {len(self.prompt_templates)} prompt templates")
        
        # Create output directory
        self.output_dir = Path(self.eval_config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")
        
        # Set random seed for reproducibility
        random.seed(self.eval_config['seed'])
        print(f"🎲 Random seed: {self.eval_config['seed']}")
        
        print("="*80 + "\n")
    
    def create_prompt(self, template: str, group0: str, group1: str, 
                     attributes: List[str]) -> str:
        """Create prompt from template with shuffled attributes"""
        # Shuffle attributes for this trial
        attrs_shuffled = attributes.copy()
        random.shuffle(attrs_shuffled)
        
        # Format template
        prompt = template.format(
            group0=group0,
            group1=group1,
            attributes=', '.join(attrs_shuffled)
        )
        
        return prompt
    
    def prepare_batch_prompts(self, stimuli: IATStimuli, variation_name: str,
                            num_iterations: int, tokenizer) -> Tuple[List[str], List[Dict]]:
        """
        Prepare a batch of prompts for a given stimuli and variation
        Returns: (formatted_prompts, metadata_list)
        """
        template = self.prompt_templates[variation_name]
        formatted_prompts = []
        metadata = []
        
        for iteration in range(num_iterations):
            # Randomly select concept exemplars
            group0 = random.choice(stimuli.concept_a_list)
            group1 = random.choice(stimuli.concept_b_list)
            
            # Randomly order groups (counterbalancing)
            if random.random() > 0.5:
                group0, group1 = group1, group0
            
            # Create raw prompt
            raw_prompt = self.create_prompt(template, group0, group1, stimuli.attributes)
            
            # Format with chat template
            messages = [{"role": "user", "content": raw_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            formatted_prompts.append(formatted_prompt)
            metadata.append({
                'category': stimuli.category,
                'dataset': stimuli.dataset,
                'variation': variation_name,
                'iteration': iteration,
                'group0': group0,
                'group1': group1,
                'attributes': ', '.join(stimuli.attributes),
                'prompt': raw_prompt  # Store raw prompt, not formatted
            })
        
        return formatted_prompts, metadata
    
    def process_model(self, model: ModelConfig) -> pd.DataFrame:
        """Process all IAT trials for a single model"""
        
        print("\n" + "="*80)
        print(f"🚀 Processing model: {model.display_name}")
        print(f"   Full name: {model.name}")
        print(f"   Tensor Parallel: {model.tensor_parallel_size}")
        print(f"   Pipeline Parallel: {model.pipeline_parallel_size}")
        print("="*80)
        
        # Load model with vLLM
        try:
            print(f"\n⏳ Loading model into vLLM...")
            llm = LLM(
                model=model.name,
                tensor_parallel_size=model.tensor_parallel_size,
                pipeline_parallel_size=model.pipeline_parallel_size,
                gpu_memory_utilization=model.gpu_memory_utilization,
                max_model_len=model.max_model_len,
                quantization=model.quantization,
                trust_remote_code=True,
                disable_log_stats=True,
            )
            tokenizer = llm.get_tokenizer()
            print(f"✅ Model loaded successfully")
        
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print(f"⏭️  Skipping {model.display_name}")
            return None
        
        # Prepare sampling parameters
        sampling_params = SamplingParams(
            temperature=self.eval_config['temperature'],
            max_tokens=self.eval_config['max_tokens']
        )
        
        # Get categories to evaluate
        categories_to_eval = self.eval_config['categories']
        variations_to_eval = self.eval_config['prompt_variations']
        num_iterations = self.eval_config['iterations']
        
        all_results = []
        
        # Loop through all categories
        for category in categories_to_eval:
            print(f"\n📊 Category: {category}")
            
            # Get all stimuli for this category
            category_stimuli = self.stimuli_loader.get_stimuli_by_category(category)
            
            if not category_stimuli:
                print(f"   ⚠️  No stimuli found for category '{category}', skipping")
                continue
            
            # Loop through datasets within this category
            for stimuli in category_stimuli:
                print(f"   📦 Dataset: {stimuli.dataset}")
                print(f"      Concepts: {stimuli.concept_a_list} vs {stimuli.concept_b_list}")
                print(f"      Attributes: {len(stimuli.attributes)} words")
                
                # Loop through prompt variations
                for variation_name in variations_to_eval:
                    print(f"      🔄 Variation: {variation_name}")
                    
                    # Prepare batch of prompts (50 iterations)
                    formatted_prompts, metadata = self.prepare_batch_prompts(
                        stimuli, variation_name, num_iterations, tokenizer
                    )
                    
                    # Generate responses (batched)
                    print(f"         Generating {len(formatted_prompts)} responses...")
                    try:
                        outputs = llm.generate(
                            prompts=formatted_prompts,
                            sampling_params=sampling_params,
                            use_tqdm=False
                        )
                        
                        # Extract results
                        for output, meta in zip(outputs, metadata):
                            response = output.outputs[0].text.strip()
                            
                            result = {
                                'model': model.display_name,
                                'temperature': self.eval_config['temperature'],
                                'category': meta['category'],
                                'dataset': meta['dataset'],
                                'variation': meta['variation'],
                                'iteration': meta['iteration'],
                                'group0': meta['group0'],
                                'group1': meta['group1'],
                                'attributes': meta['attributes'],
                                'prompt': meta['prompt'],
                                'response': response
                            }
                            all_results.append(result)
                        
                        print(f"         ✅ Completed {len(outputs)} responses")
                    
                    except Exception as e:
                        print(f"         ❌ Error during generation: {e}")
                        continue
        
        # Cleanup
        print(f"\n🧹 Cleaning up model...")
        del llm
        torch.cuda.empty_cache()
        print(f"✅ Model unloaded")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        output_file = self.output_dir / f"{model.display_name}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        print(f"   Total responses: {len(results_df)}")
        
        return results_df
    
    def run(self):
        """Run benchmark on all enabled models"""
        
        # Get enabled models sorted by size (small to large)
        models = self.config.get_enabled_models()
        
        # Sort by total GPUs needed (ascending)
        models_sorted = sorted(models, key=lambda m: m.total_gpus_needed)
        
        print(f"\n🎯 Found {len(models_sorted)} enabled models")
        for i, model in enumerate(models_sorted, 1):
            print(f"   {i}. {model.display_name} ({model.total_gpus_needed} GPUs)")
        
        # Validate GPU requirements
        validation = self.config.validate_gpu_requirements()
        if not validation['valid']:
            print("\n❌ GPU validation failed:")
            for error in validation['errors']:
                print(f"   - {error}")
            return
        
        if validation['warnings']:
            print("\n⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"   - {warning}")
        
        # Process each model sequentially
        print(f"\n{'='*80}")
        print(f"Starting benchmark...")
        print(f"{'='*80}\n")
        
        results_summary = []
        
        for i, model in enumerate(models_sorted, 1):
            print(f"\n{'#'*80}")
            print(f"# Model {i}/{len(models_sorted)}")
            print(f"{'#'*80}")
            
            result_df = self.process_model(model)
            
            if result_df is not None:
                results_summary.append({
                    'model': model.display_name,
                    'status': 'completed',
                    'num_responses': len(result_df)
                })
            else:
                results_summary.append({
                    'model': model.display_name,
                    'status': 'failed',
                    'num_responses': 0
                })
        
        # Print final summary
        print("\n" + "="*80)
        print("BENCHMARK COMPLETE")
        print("="*80)
        print("\nResults Summary:")
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False))
        print(f"\n📁 All results saved to: {self.output_dir}")
        print("="*80)


def main():
    """Main entry point"""
    benchmark = IATBenchmark("config/models_config.yml")
    benchmark.run()


if __name__ == "__main__":
    main()