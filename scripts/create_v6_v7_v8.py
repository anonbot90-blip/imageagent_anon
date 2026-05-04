#!/usr/bin/env python3
"""
Create v6-v8 variation files with 500 prompts each
v6: Time Period/Era emphasis
v7: Artistic Medium emphasis
v8: Atmospheric Effects emphasis
"""

import json
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def apply_time_period_variation(base_prompt, new_id):
    """Apply v6 time period/era variation"""
    # Era modifiers
    era_modifiers = [
        ("with 1950s retro aesthetic and mid-century design", "featuring 1950s vintage style and period details"),
        ("in Victorian 1800s style with ornate details", "transformed to Victorian era with historical elements"),
        ("with medieval dark ages atmosphere", "featuring medieval period architecture and props"),
        ("in far future 2200 style with advanced technology", "transformed to far future with sci-fi elements"),
        ("with cyberpunk dystopia aesthetic", "featuring cyberpunk era technology and neon"),
        ("in ancient classical style", "transformed to ancient times with historical accuracy"),
        ("with steampunk alternate history design", "featuring steampunk Victorian-tech fusion"),
        ("in Art Deco 1920s style", "transformed to 1920s with geometric patterns"),
        ("with Renaissance 1500s aesthetic", "featuring Renaissance period art and architecture"),
        ("in near future 2050 style", "transformed to near future with emerging technology"),
    ]
    
    modifier_idx = (new_id - 2501) % len(era_modifiers)
    gen_modifier, edit_modifier = era_modifiers[modifier_idx]
    
    new_prompt = base_prompt.copy()
    new_prompt['id'] = new_id
    new_prompt['name'] = f"{base_prompt['name']} (Era Focus)"
    new_prompt['generation'] = base_prompt['generation'].copy()
    new_prompt['edit'] = base_prompt['edit'].copy()
    
    # Add era emphasis
    new_prompt['generation']['text'] = f"{base_prompt['generation']['text']}, {gen_modifier}"
    new_prompt['edit']['text'] = f"{base_prompt['edit']['text']}, {edit_modifier}"
    
    return new_prompt


def apply_artistic_medium_variation(base_prompt, new_id):
    """Apply v7 artistic medium variation"""
    # Artistic medium modifiers
    medium_modifiers = [
        ("rendered as oil painting with thick brushstrokes", "transformed to oil painting with visible canvas texture"),
        ("in watercolor style with flowing washes", "rendered as watercolor with soft blended colors"),
        ("as anime cel-shaded illustration", "transformed to anime style with vibrant cel-shading"),
        ("in pixel art retro style", "rendered as pixel art with retro gaming aesthetic"),
        ("as pencil sketch with hatching", "transformed to pencil drawing with sketch lines"),
        ("rendered as digital painting", "transformed to digital art with painting techniques"),
        ("in charcoal sketch style", "rendered as charcoal drawing with dramatic shading"),
        ("as vector art with clean lines", "transformed to vector illustration with geometric shapes"),
        ("in 3D rendered style", "rendered as 3D computer graphics"),
        ("as cartoon illustration", "transformed to cartoon style with simplified forms"),
    ]
    
    modifier_idx = (new_id - 3001) % len(medium_modifiers)
    gen_modifier, edit_modifier = medium_modifiers[modifier_idx]
    
    new_prompt = base_prompt.copy()
    new_prompt['id'] = new_id
    new_prompt['name'] = f"{base_prompt['name']} (Artistic Medium)"
    new_prompt['generation'] = base_prompt['generation'].copy()
    new_prompt['edit'] = base_prompt['edit'].copy()
    
    # Add artistic medium emphasis
    new_prompt['generation']['text'] = f"{base_prompt['generation']['text']}, {gen_modifier}"
    new_prompt['edit']['text'] = f"{base_prompt['edit']['text']}, {edit_modifier}"
    
    return new_prompt


def apply_atmospheric_effects_variation(base_prompt, new_id):
    """Apply v8 atmospheric effects variation"""
    # Atmospheric effects modifiers
    effects_modifiers = [
        ("with magical sparkles floating in air", "enhanced with magical particle effects"),
        ("featuring volumetric fog effects", "transformed with atmospheric fog layers"),
        ("with dust particles in light beams", "enhanced with dust motes and god rays"),
        ("featuring soft bokeh lights", "transformed with bokeh light effects"),
        ("with glowing aura effects", "enhanced with ethereal glow and aura"),
        ("featuring lens flare and light streaks", "transformed with cinematic lens effects"),
        ("with smoke and haze atmosphere", "enhanced with atmospheric smoke effects"),
        ("featuring floating embers", "transformed with glowing ember particles"),
        ("with bioluminescent glow", "enhanced with bioluminescent lighting effects"),
        ("featuring holographic glitch effects", "transformed with digital glitch artifacts"),
    ]
    
    modifier_idx = (new_id - 3501) % len(effects_modifiers)
    gen_modifier, edit_modifier = effects_modifiers[modifier_idx]
    
    new_prompt = base_prompt.copy()
    new_prompt['id'] = new_id
    new_prompt['name'] = f"{base_prompt['name']} (Atmospheric Effects)"
    new_prompt['generation'] = base_prompt['generation'].copy()
    new_prompt['edit'] = base_prompt['edit'].copy()
    
    # Add atmospheric effects emphasis
    new_prompt['generation']['text'] = f"{base_prompt['generation']['text']}, {gen_modifier}"
    new_prompt['edit']['text'] = f"{base_prompt['edit']['text']}, {edit_modifier}"
    
    return new_prompt


def create_variation_file(base_prompts, variation_name, variation_func, start_id, output_file, variation_key):
    """Create a new variation file with 500 prompts"""
    
    print(f"\n{'='*80}")
    print(f"Creating {variation_name}")
    print(f"{'='*80}")
    
    # Apply variation to all 500 base prompts
    new_prompts = []
    for i, base_prompt in enumerate(base_prompts):
        new_id = start_id + i
        new_prompt = variation_func(base_prompt, new_id)
        new_prompts.append(new_prompt)
    
    # Create new variation file structure
    variation_data = {
        "version": "2.0",
        "description": f"Variation: 500 prompts with {variation_name.lower()} emphasis",
        "total_prompts": 500,
        "variation": variation_key,
        "prompts": new_prompts
    }
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(variation_data, f, indent=2)
    
    print(f"✅ Generated {len(new_prompts)} prompts (IDs {start_id}-{start_id + len(new_prompts) - 1})")
    print(f"✅ Saved to {output_file.name}")
    
    return new_prompts


def main():
    """Main function"""
    
    print("="*80)
    print("CREATING V6-V8 VARIATIONS WITH 500 PROMPTS EACH")
    print("="*80)
    print()
    
    # Load base prompts (now 500)
    base_file = PROJECT_ROOT / "config" / "prompt_theme_100.json"
    with open(base_file, 'r') as f:
        base_data = json.load(f)
    
    base_prompts = base_data['prompts']
    print(f"✓ Loaded {len(base_prompts)} base prompts from {base_file.name}")
    
    if len(base_prompts) != 500:
        print(f"❌ Error: Expected 500 base prompts, found {len(base_prompts)}")
        sys.exit(1)
    
    # Create new variations
    variations = [
        ("v6 (Time Period/Era)", apply_time_period_variation, 2501, "prompt_theme_100_v6.json", "time_period_era"),
        ("v7 (Artistic Medium)", apply_artistic_medium_variation, 3001, "prompt_theme_100_v7.json", "artistic_medium"),
        ("v8 (Atmospheric Effects)", apply_atmospheric_effects_variation, 3501, "prompt_theme_100_v8.json", "atmospheric_effects"),
    ]
    
    for var_name, var_func, start_id, filename, var_key in variations:
        output_file = PROJECT_ROOT / "config" / filename
        create_variation_file(base_prompts, var_name, var_func, start_id, output_file, var_key)
    
    print(f"\n{'='*80}")
    print("✅ ALL NEW VARIATIONS CREATED SUCCESSFULLY!")
    print(f"{'='*80}")
    print()
    print("Summary:")
    print(f"  - v6 (Time Period/Era): IDs 2501-3000 (500 prompts)")
    print(f"  - v7 (Artistic Medium): IDs 3001-3500 (500 prompts)")
    print(f"  - v8 (Atmospheric Effects): IDs 3501-4000 (500 prompts)")
    print(f"  - Total: 1500 new prompts across v6-v8")
    print()
    print("Grand Total Across All Variations:")
    print(f"  - Base (v1): 1-500 (500 prompts)")
    print(f"  - v2-v5: 501-2500 (2000 prompts)")
    print(f"  - v6-v8: 2501-4000 (1500 prompts)")
    print(f"  - TOTAL: 4000 unique prompts!")


if __name__ == "__main__":
    main()

