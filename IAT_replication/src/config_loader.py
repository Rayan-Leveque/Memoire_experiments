#!/usr/bin/env python3
"""
config_loader.py
Loads and validates the models_config.yml configuration file
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Represents a single model configuration"""
    name: str
    display_name: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    enabled: bool
    revision: str = None
    gpu_memory_utilization: float = None
    max_model_len: int = None
    quantization: str = None
    notes: str = ""
    
    @property
    def total_gpus_needed(self) -> int:
        """Calculate total GPUs needed (TP × PP)"""
        return self.tensor_parallel_size * self.pipeline_parallel_size
    
    def __str__(self):
        return (f"{self.display_name}: "
                f"TP={self.tensor_parallel_size}, "
                f"PP={self.pipeline_parallel_size}, "
                f"Total GPUs={self.total_gpus_needed}")


class ConfigLoader:
    """Loads and validates YAML configuration"""
    
    def __init__(self, config_path: str = "config/models_config.yml"):
        self.config_path = Path(config_path)
        self.config = self._load_yaml()
        self._validate_config()
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate_config(self):
        """Validate configuration structure"""
        required_sections = ['hardware', 'evaluation', 'models']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
        
        if not self.config['models']:
            raise ValueError("No models defined in configuration")
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration"""
        return self.config['hardware']
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation settings"""
        return self.config['evaluation']
    
    def get_all_models(self) -> List[ModelConfig]:
        """Get all model configurations"""
        models = []
        eval_config = self.get_evaluation_config()
        
        for model_dict in self.config['models']:
            # Use defaults from evaluation config if not specified
            model = ModelConfig(
                name=model_dict['name'],
                display_name=model_dict['display_name'],
                tensor_parallel_size=model_dict['tensor_parallel_size'],
                pipeline_parallel_size=model_dict['pipeline_parallel_size'],
                enabled=model_dict.get('enabled', True),
                revision=model_dict.get('revision'),
                gpu_memory_utilization=model_dict.get('gpu_memory_utilization', 
                                                      eval_config['gpu_memory_utilization']),
                max_model_len=model_dict.get('max_model_len', 
                                             eval_config['max_model_len']),
                quantization=model_dict.get('quantization'),
                notes=model_dict.get('notes', '')
            )
            models.append(model)
        
        return models
    
    def get_enabled_models(self) -> List[ModelConfig]:
        """Get only enabled models"""
        return [m for m in self.get_all_models() if m.enabled]
    
    def get_models_by_size(self) -> Dict[str, List[ModelConfig]]:
        """Group enabled models by GPU requirements"""
        models = self.get_enabled_models()
        
        grouped = {
            'small': [],   # 1 GPU
            'medium': [],  # 2 GPUs
            'large': [],   # 3+ GPUs
        }
        
        for model in models:
            if model.total_gpus_needed == 1:
                grouped['small'].append(model)
            elif model.total_gpus_needed == 2:
                grouped['medium'].append(model)
            else:
                grouped['large'].append(model)
        
        return grouped
    
    def get_prompt_templates(self) -> Dict[str, str]:
        """Get prompt templates"""
        templates = {}
        for name, template_config in self.config.get('prompt_templates', {}).items():
            templates[name] = template_config['template']
        return templates
    
    def get_categories(self) -> List[str]:
        """Get evaluation categories"""
        return self.get_evaluation_config()['categories']
    
    def get_prompt_variations(self) -> List[str]:
        """Get prompt variation names"""
        return self.get_evaluation_config()['prompt_variations']
    
    def validate_gpu_requirements(self) -> Dict[str, Any]:
        """Validate that all enabled models can run on available hardware"""
        hardware = self.get_hardware_config()
        num_gpus = hardware['num_gpus']
        models = self.get_enabled_models()
        
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'summary': {}
        }
        
        for model in models:
            if model.total_gpus_needed > num_gpus:
                validation['valid'] = False
                validation['errors'].append(
                    f"{model.display_name} needs {model.total_gpus_needed} GPUs "
                    f"but only {num_gpus} available"
                )
            elif model.total_gpus_needed == num_gpus:
                validation['warnings'].append(
                    f"{model.display_name} will use all {num_gpus} GPUs "
                    f"(cannot run in parallel with other models)"
                )
        
        # Summary
        grouped = self.get_models_by_size()
        validation['summary'] = {
            'total_models': len(models),
            'small_models': len(grouped['small']),
            'medium_models': len(grouped['medium']),
            'large_models': len(grouped['large']),
            'available_gpus': num_gpus
        }
        
        return validation
    
    def print_summary(self):
        """Print configuration summary"""
        print("="*80)
        print("Configuration Summary")
        print("="*80)
        
        # Hardware
        hardware = self.get_hardware_config()
        print(f"\nHardware:")
        print(f"  GPUs: {hardware['num_gpus']}x {hardware['gpu_model']}")
        print(f"  VRAM per GPU: {hardware['vram_per_gpu']}GB")
        print(f"  Total VRAM: {hardware['total_vram']}GB")
        
        # Evaluation
        eval_config = self.get_evaluation_config()
        print(f"\nEvaluation:")
        print(f"  Iterations: {eval_config['iterations']}")
        print(f"  Categories: {', '.join(eval_config['categories'])}")
        print(f"  Prompt variations: {', '.join(eval_config['prompt_variations'])}")
        
        # Models
        models = self.get_enabled_models()
        grouped = self.get_models_by_size()
        
        print(f"\nModels:")
        print(f"  Total enabled: {len(models)}")
        print(f"  Small (1 GPU): {len(grouped['small'])}")
        for model in grouped['small']:
            print(f"    - {model.display_name}")
        
        print(f"  Medium (2 GPUs): {len(grouped['medium'])}")
        for model in grouped['medium']:
            print(f"    - {model.display_name}")
        
        print(f"  Large (3+ GPUs): {len(grouped['large'])}")
        for model in grouped['large']:
            print(f"    - {model.display_name} ({model.total_gpus_needed} GPUs)")
        
        # Validation
        validation = self.validate_gpu_requirements()
        print(f"\nValidation:")
        if validation['valid']:
            print("  ✓ All enabled models can run on available hardware")
        else:
            print("  ✗ Some models cannot run:")
            for error in validation['errors']:
                print(f"    - {error}")
        
        if validation['warnings']:
            print("  Warnings:")
            for warning in validation['warnings']:
                print(f"    - {warning}")
        
        print("="*80)


def main():
    """Example usage"""
    # Load configuration
    config = ConfigLoader("config/models_config.yml")
    
    # Print summary
    config.print_summary()
    
    # Get specific configurations
    print("\n" + "="*80)
    print("Example: Getting specific configurations")
    print("="*80)
    
    # Get all enabled models
    models = config.get_enabled_models()
    print(f"\nEnabled models: {len(models)}")
    for model in models[:3]:  # Show first 3
        print(f"  - {model}")
    
    # Get prompt templates
    templates = config.get_prompt_templates()
    print(f"\nPrompt templates: {list(templates.keys())}")
    
    # Get evaluation settings
    eval_settings = config.get_evaluation_config()
    print(f"\nIterations: {eval_settings['iterations']}")
    print(f"Temperature: {eval_settings['temperature']}")


if __name__ == "__main__":
    main()
