#!/usr/bin/env python3
"""
Expand v2-v5 variation files from 100 to 500 prompts each
Applies variation emphasis to all 500 base prompts
"""

import json
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def apply_seasonal_weather_variation(base_prompt, new_id):
    """Apply v2 seasonal/weather variation"""
    # Weather/season modifiers
    weather_seasons = [
        ("hot summer afternoon with hazy heat", "maintaining summer noon atmosphere"),
        ("golden autumn sunset with warm light", "maintaining autumn evening atmosphere"),
        ("frosty winter night with moonlight", "maintaining winter night atmosphere"),
        ("fresh spring morning with dew", "maintaining spring morning atmosphere"),
        ("cool autumn rain with falling leaves", "maintaining autumn rain atmosphere"),
        ("bright summer midday with clear skies", "maintaining summer midday atmosphere"),
        ("cold winter dawn with frost", "maintaining winter dawn atmosphere"),
        ("warm spring afternoon with gentle breeze", "maintaining spring afternoon atmosphere"),
        ("misty autumn evening with fog", "maintaining misty autumn atmosphere"),
        ("snowy winter day with soft flakes", "maintaining snowy winter atmosphere"),
    ]
    
    modifier_idx = (new_id - 501) % len(weather_seasons)
    gen_modifier, edit_modifier = weather_seasons[modifier_idx]
    
    new_prompt = base_prompt.copy()
    new_prompt['id'] = new_id
    new_prompt['name'] = f"{base_prompt['name']} (Seasonal/Weather)"
    new_prompt['generation'] = base_prompt['generation'].copy()
    new_prompt['edit'] = base_prompt['edit'].copy()
    
    # Add weather/season to generation text
    new_prompt['generation']['text'] = f"{base_prompt['generation']['text']} {gen_modifier}"
    
    # Add weather/season to edit text
    new_prompt['edit']['text'] = f"{base_prompt['edit']['text']}, {edit_modifier}"
    
    return new_prompt


def apply_architectural_variation(base_prompt, new_id):
    """Apply v3 architectural focus variation"""
    # Architectural modifiers
    arch_styles = [
        ("featuring minimalist modern architectural design", "highlighting metallic and glass surfaces"),
        ("showcasing brutalist concrete architecture", "showcasing weathered brick and aged materials"),
        ("with Victorian architectural elements and intricate details", "featuring polished marble and luxury finishes"),
        ("displaying Art Deco geometric patterns", "emphasizing ornate decorative elements"),
        ("featuring Gothic architectural details", "highlighting stone craftsmanship and arches"),
        ("showcasing industrial warehouse aesthetics", "emphasizing exposed structural elements"),
        ("with Japanese traditional architecture", "featuring wooden construction and natural materials"),
        ("displaying Renaissance classical proportions", "highlighting symmetrical design and columns"),
        ("featuring futuristic sleek architecture", "emphasizing curved surfaces and innovation"),
        ("showcasing organic biomorphic forms", "highlighting natural flowing shapes"),
    ]
    
    modifier_idx = (new_id - 1001) % len(arch_styles)
    gen_modifier, edit_modifier = arch_styles[modifier_idx]
    
    new_prompt = base_prompt.copy()
    new_prompt['id'] = new_id
    new_prompt['name'] = f"{base_prompt['name']} (Architectural Focus)"
    new_prompt['generation'] = base_prompt['generation'].copy()
    new_prompt['edit'] = base_prompt['edit'].copy()
    
    # Add architectural emphasis
    new_prompt['generation']['text'] = f"{base_prompt['generation']['text']}, {gen_modifier}"
    new_prompt['edit']['text'] = f"{base_prompt['edit']['text']}, {edit_modifier}"
    
    return new_prompt


def apply_activity_density_variation(base_prompt, new_id):
    """Apply v4 activity/population density variation"""
    # Activity/density modifiers
    activity_levels = [
        ("with sparse population and quiet atmosphere", "centered around work and productivity"),
        ("completely deserted and abandoned", "featuring cultural and artistic activities"),
        ("moderately populated with casual activity", "showcasing sports and physical activities"),
        ("bustling with crowds and energy", "highlighting social gatherings and celebrations"),
        ("with minimal human presence", "emphasizing solitary contemplation"),
        ("filled with busy activity", "featuring commercial transactions"),
        ("showing peaceful solitude", "highlighting creative endeavors"),
        ("crowded with diverse people", "showcasing entertainment and leisure"),
        ("with scattered individuals", "emphasizing educational activities"),
        ("densely packed with activity", "featuring festive celebrations"),
    ]
    
    modifier_idx = (new_id - 1501) % len(activity_levels)
    gen_modifier, edit_modifier = activity_levels[modifier_idx]
    
    new_prompt = base_prompt.copy()
    new_prompt['id'] = new_id
    new_prompt['name'] = f"{base_prompt['name']} (Activity Variation)"
    new_prompt['generation'] = base_prompt['generation'].copy()
    new_prompt['edit'] = base_prompt['edit'].copy()
    
    # Add activity/density emphasis
    new_prompt['generation']['text'] = f"{base_prompt['generation']['text']}, {gen_modifier}"
    new_prompt['edit']['text'] = f"{base_prompt['edit']['text']}, {edit_modifier}"
    
    return new_prompt


def apply_lighting_color_variation(base_prompt, new_id):
    """Apply v5 lighting/color palette variation"""
    # Lighting/color modifiers
    lighting_colors = [
        ("illuminated by cool blue twilight", "using muted desaturated tones"),
        ("lit by dramatic side lighting and shadows", "featuring high contrast black and white emphasis"),
        ("glowing with soft diffused overcast light", "showcasing warm earth tone palette"),
        ("bathed in warm golden hour light", "using vibrant saturated colors"),
        ("under harsh midday sun", "featuring stark high contrast"),
        ("with moody atmospheric lighting", "emphasizing deep rich colors"),
        ("illuminated by soft morning light", "using pastel color palette"),
        ("lit by dramatic backlighting", "featuring silhouette emphasis"),
        ("glowing with neon colored lights", "showcasing electric color scheme"),
        ("under soft candlelight", "using warm amber tones"),
    ]
    
    modifier_idx = (new_id - 2001) % len(lighting_colors)
    gen_modifier, edit_modifier = lighting_colors[modifier_idx]
    
    new_prompt = base_prompt.copy()
    new_prompt['id'] = new_id
    new_prompt['name'] = f"{base_prompt['name']} (Lighting/Color Variation)"
    new_prompt['generation'] = base_prompt['generation'].copy()
    new_prompt['edit'] = base_prompt['edit'].copy()
    
    # Add lighting/color emphasis
    new_prompt['generation']['text'] = f"{base_prompt['generation']['text']}, {gen_modifier}"
    new_prompt['edit']['text'] = f"{base_prompt['edit']['text']}, {edit_modifier}"
    
    return new_prompt


def expand_variation_file(base_prompts, variation_name, variation_func, start_id, output_file):
    """Expand a variation file from 100 to 500 prompts"""
    
    print(f"\n{'='*80}")
    print(f"Expanding {variation_name}")
    print(f"{'='*80}")
    
    # Apply variation to all 500 base prompts
    new_prompts = []
    for i, base_prompt in enumerate(base_prompts):
        new_id = start_id + i
        new_prompt = variation_func(base_prompt, new_id)
        new_prompts.append(new_prompt)
    
    # Load existing variation file to preserve metadata
    with open(output_file, 'r') as f:
        existing_data = json.load(f)
    
    # Update with new prompts
    existing_data['prompts'] = new_prompts
    existing_data['total_prompts'] = 500
    existing_data['description'] = existing_data['description'].replace("100", "500")
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"✅ Generated {len(new_prompts)} prompts (IDs {start_id}-{start_id + len(new_prompts) - 1})")
    print(f"✅ Saved to {output_file.name}")
    
    return new_prompts


def main():
    """Main function"""
    
    print("="*80)
    print("EXPANDING V2-V5 VARIATIONS FROM 100 TO 500 PROMPTS EACH")
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
    
    # Expand each variation
    variations = [
        ("v2 (Seasonal/Weather)", apply_seasonal_weather_variation, 501, "prompt_theme_100_v2.json"),
        ("v3 (Architectural Focus)", apply_architectural_variation, 1001, "prompt_theme_100_v3.json"),
        ("v4 (Activity Density)", apply_activity_density_variation, 1501, "prompt_theme_100_v4.json"),
        ("v5 (Lighting/Color)", apply_lighting_color_variation, 2001, "prompt_theme_100_v5.json"),
    ]
    
    for var_name, var_func, start_id, filename in variations:
        output_file = PROJECT_ROOT / "config" / filename
        expand_variation_file(base_prompts, var_name, var_func, start_id, output_file)
    
    print(f"\n{'='*80}")
    print("✅ ALL VARIATIONS EXPANDED SUCCESSFULLY!")
    print(f"{'='*80}")
    print()
    print("Summary:")
    print(f"  - v2 (Seasonal/Weather): IDs 501-1000 (500 prompts)")
    print(f"  - v3 (Architectural Focus): IDs 1001-1500 (500 prompts)")
    print(f"  - v4 (Activity Density): IDs 1501-2000 (500 prompts)")
    print(f"  - v5 (Lighting/Color): IDs 2001-2500 (500 prompts)")
    print(f"  - Total: 2000 prompts across v2-v5")


if __name__ == "__main__":
    main()

