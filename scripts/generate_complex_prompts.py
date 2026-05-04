#!/usr/bin/env python3
"""
Generate 400 complex prompts for unified multi-constraint dataset generation.

Distribution:
- Level 1 (Dual): 120 prompts (30%)
- Level 2 (Transform + Constraint): 160 prompts (40%)
- Level 3 (Multi-Constraint): 80 prompts (20%)
- Level 4 (Conditional): 40 prompts (10%)
"""

import json
import random
from pathlib import Path

# Base scenes for generation
BASE_SCENES = [
    {"theme": "modern_living_room", "text": "A modern living room with large windows, minimalist furniture, LED lighting, and neutral color palette"},
    {"theme": "traditional_bedroom", "text": "A traditional bedroom with queen bed, wooden nightstands, table lamp, and soft ambient lighting"},
    {"theme": "urban_kitchen", "text": "An urban kitchen with stainless steel appliances, white cabinets, marble countertops, and pendant lights"},
    {"theme": "cozy_office", "text": "A cozy home office with desk, computer, bookshelf, desk lamp, and natural daylight from window"},
    {"theme": "minimalist_bathroom", "text": "A minimalist bathroom with walk-in shower, floating vanity, large mirror, and bright overhead lighting"},
    {"theme": "rustic_dining_room", "text": "A rustic dining room with wooden table, chairs, chandelier, and large windows showing garden view"},
    {"theme": "contemporary_studio", "text": "A contemporary studio apartment with open layout, modern furniture, track lighting, and city view"},
    {"theme": "classic_library", "text": "A classic library with floor-to-ceiling bookshelves, reading chair, desk lamp, and warm wooden tones"},
    {"theme": "industrial_loft", "text": "An industrial loft with exposed brick walls, high ceilings, metal fixtures, and large warehouse windows"},
    {"theme": "zen_meditation_room", "text": "A zen meditation room with minimal furniture, floor cushions, bamboo elements, and soft natural light"},
]

# Transformation types
DUAL_TRANSFORMS = [
    {
        "type1": "theme_change",
        "target1": "Victorian era with ornate furniture and wallpaper",
        "type2": "lighting_change",
        "target2": "warm candlelit atmosphere with soft shadows",
        "combined_text": "Transform to Victorian era with ornate furniture and wallpaper AND add warm candlelit atmosphere with soft shadows"
    },
    {
        "type1": "weather_change",
        "target1": "heavy snowfall with snow accumulation on surfaces",
        "type2": "time_change",
        "target2": "late evening blue hour lighting",
        "combined_text": "Add heavy snowfall with snow accumulation AND change time to late evening blue hour lighting"
    },
    {
        "type1": "style_change",
        "target1": "Art Deco geometric patterns and gold accents",
        "type2": "color_change",
        "target2": "jewel-tone color palette with emerald and sapphire hues",
        "combined_text": "Transform to Art Deco style with geometric patterns AND apply jewel-tone color palette with emerald and sapphire hues"
    },
]

CONSTRAINT_TRANSFORMS = [
    {
        "transform": "Transform to industrial warehouse aesthetic with exposed elements",
        "constraint": "preserve the wooden furniture",
        "reason": "maintain warmth and contrast with industrial setting"
    },
    {
        "transform": "Make it heavy rainy day with wet surfaces and water droplets",
        "constraint": "preserve the bright sunny lighting indoors",
        "reason": "show contrast between outdoor weather and indoor brightness"
    },
    {
        "transform": "Change to futuristic high-tech environment with holographic displays",
        "constraint": "keep all plants and natural elements",
        "reason": "maintain organic contrast with synthetic technology"
    },
]

MULTI_CONSTRAINTS = [
    {
        "transform": "Transform to space station interior",
        "constraint1": "keep the bed as-is",
        "constraint2": "add zero-gravity floating objects",
        "combined": "Transform to space station interior, preserve the existing bed exactly, AND add zero-gravity floating objects and effects"
    },
    {
        "transform": "Change to 1920s Art Deco ballroom",
        "constraint1": "preserve modern electronics",
        "constraint2": "add golden hour lighting",
        "combined": "Transform to 1920s Art Deco ballroom, keep all modern electronics visible, AND add warm golden hour lighting through windows"
    },
]

CONDITIONAL_TRANSFORMS = [
    {
        "condition": "IF there's a window",
        "then_action": "show heavy rain outside with wet glass",
        "else_action": "add indoor potted plants for freshness",
        "combined": "IF there's a window, show heavy rain outside with wet glass; OTHERWISE add indoor potted plants along walls"
    },
    {
        "condition": "IF outdoor area is visible",
        "then_action": "add dense fog effect outdoors only",
        "else_action": "keep current indoor atmosphere",
        "combined": "IF outdoor area is visible, add dense fog effect outdoors only; OTHERWISE keep current indoor atmosphere unchanged"
    },
]


def generate_level1_prompts(start_id=1, count=120):
    """Generate Level 1: Dual transformation prompts"""
    prompts = []
    
    # Expand transformation combinations
    all_combinations = [
        ("Victorian era", "candlelit warm lighting"),
        ("industrial loft", "dramatic spotlight from above"),
        ("space station", "cool LED blue lighting"),
        ("medieval castle", "torch-lit flickering ambiance"),
        ("Art Deco 1920s", "golden hour sunset glow"),
        ("cyberpunk city", "neon pink and blue lighting"),
        ("ancient temple", "mystical green fog atmosphere"),
        ("winter wonderland", "soft blue twilight mood"),
        ("autumn forest", "warm orange afternoon light"),
        ("tropical beach", "bright midday overhead sun"),
        ("Art Nouveau 1900s", "stained glass colorful lighting"),
        ("steampunk workshop", "brass warm gas lamp lighting"),
        ("underwater base", "cool aqua bioluminescent glow"),
        ("desert oasis", "harsh bright desert sunlight"),
        ("haunted mansion", "eerie green moonlit atmosphere"),
    ]
    
    idx = start_id
    for i in range(count):
        scene = BASE_SCENES[i % len(BASE_SCENES)]
        combo = all_combinations[i % len(all_combinations)]
        
        prompts.append({
            "id": idx,
            "name": f"Dual: {combo[0]} + {combo[1]}",
            "category": "level_1_dual_transformation",
            "complexity_level": 1,
            "text": scene["text"],
            "theme": scene["theme"],
            "model": "fast",
            "resolution": "1024x1024",
            "edit_info": {
                "text": f"Transform to {combo[0]} AND add {combo[1]}",
                "transformations": [
                    {
                        "type": "theme_change",
                        "target": combo[0],
                        "priority": 1
                    },
                    {
                        "type": "lighting_change",
                        "target": combo[1],
                        "priority": 2
                    }
                ],
                "expected_actions": ["architecture_style", "mood_lighting"],
                "complexity_tags": ["dual_transformation", "no_constraints"]
            },
            "_cycle": 0,
            "_original_id": idx
        })
        idx += 1
    
    return prompts


def generate_level2_prompts(start_id=121, count=160):
    """Generate Level 2: Transform + Constraint prompts"""
    prompts = []
    
    constraint_patterns = [
        ("Victorian era with ornate details", "preserve the modern LED lighting fixtures", "maintain technological contrast"),
        ("industrial warehouse with exposed brick", "keep all wooden furniture", "preserve warmth in industrial setting"),
        ("Art Deco ballroom with geometric patterns", "preserve contemporary technology devices", "show anachronistic contrast"),
        ("medieval castle interior", "keep modern kitchen appliances", "create temporal paradox aesthetic"),
        ("futuristic space station", "preserve all plants and greenery", "maintain organic life in synthetic environment"),
        ("1950s mid-century modern", "keep current smart home devices", "blend retro with modern tech"),
        ("Japanese traditional zen", "preserve Western-style furniture", "cultural fusion aesthetic"),
        ("Gothic cathedral interior", "keep modern glass and metal elements", "architectural time blend"),
        ("tropical beach resort", "preserve winter snow elements visible", "impossible climate combination"),
        ("underwater observatory", "keep land-based plants alive", "surreal biological impossibility"),
    ]
    
    weather_constraints = [
        ("heavy rain with wet surfaces outside", "preserve bright sunny lighting indoors", "weather isolation"),
        ("dense fog covering everything", "keep indoor areas completely clear", "fog boundary control"),
        ("heavy snowstorm blizzard", "preserve summer vegetation outside", "seasonal impossibility"),
        ("sandstorm with reduced visibility", "keep interior crystal clear", "environmental separation"),
        ("thunderstorm with dark clouds", "maintain golden hour warm lighting", "lighting contradiction"),
    ]
    
    idx = start_id
    for i in range(count):
        scene = BASE_SCENES[i % len(BASE_SCENES)]
        
        if i < count // 2:
            # Theme + preservation constraint
            pattern = constraint_patterns[i % len(constraint_patterns)]
            prompts.append({
                "id": idx,
                "name": f"Constraint: {pattern[0].split()[0]} + Preserve",
                "category": "level_2_transform_constraint",
                "complexity_level": 2,
                "text": scene["text"],
                "theme": scene["theme"],
                "model": "fast",
                "resolution": "1024x1024",
                "edit_info": {
                    "text": f"Transform to {pattern[0]}, BUT {pattern[1]}",
                    "transformations": [
                        {
                            "type": "theme_change",
                            "target": pattern[0],
                            "priority": 1
                        }
                    ],
                    "constraints": [
                        {
                            "type": "preservation",
                            "target": pattern[1],
                            "reason": pattern[2]
                        }
                    ],
                    "expected_actions": ["architecture_style", "preserve_attribute"],
                    "complexity_tags": ["single_transformation", "preservation_constraint"]
                },
                "_cycle": 0,
                "_original_id": idx
            })
        else:
            # Weather + spatial constraint
            pattern = weather_constraints[i % len(weather_constraints)]
            prompts.append({
                "id": idx,
                "name": f"Spatial: {pattern[0].split()[0]} Outside Only",
                "category": "level_2_transform_constraint",
                "complexity_level": 2,
                "text": scene["text"],
                "theme": scene["theme"],
                "model": "fast",
                "resolution": "1024x1024",
                "edit_info": {
                    "text": f"Add {pattern[0]}, BUT {pattern[1]}",
                    "transformations": [
                        {
                            "type": "weather_change",
                            "target": pattern[0],
                            "priority": 1
                        }
                    ],
                    "constraints": [
                        {
                            "type": "spatial",
                            "target": pattern[1],
                            "reason": pattern[2]
                        }
                    ],
                    "expected_actions": ["weather_conditions", "spatial_constraint"],
                    "complexity_tags": ["weather_transformation", "spatial_constraint"]
                },
                "_cycle": 0,
                "_original_id": idx
            })
        
        idx += 1
    
    return prompts


def generate_level3_prompts(start_id=281, count=80):
    """Generate Level 3: Multi-constraint composition prompts"""
    prompts = []
    
    multi_patterns = [
        {
            "transform": "Victorian Gothic cathedral interior",
            "preserve1": "modern pendant lighting",
            "preserve2": "contemporary furniture",
            "add": "mystical purple fog atmosphere",
            "text": "Transform to Victorian Gothic cathedral interior, preserve modern pendant lighting, keep contemporary furniture, AND add mystical purple fog atmosphere throughout"
        },
        {
            "transform": "futuristic cyberpunk nightclub",
            "preserve1": "traditional wooden elements",
            "preserve2": "natural plants and greenery",
            "add": "neon pink and blue dramatic lighting",
            "text": "Transform to futuristic cyberpunk nightclub, preserve all traditional wooden elements, keep natural plants visible, AND add neon pink and blue dramatic lighting"
        },
        {
            "transform": "underwater research station",
            "preserve1": "land-based furniture arrangement",
            "preserve2": "warm ambient lighting",
            "add": "aquatic creatures swimming past windows",
            "text": "Transform to underwater research station with porthole windows, preserve land-based furniture arrangement exactly, maintain warm ambient lighting, AND add aquatic creatures swimming past windows"
        },
        {
            "transform": "Art Deco 1920s luxury train car",
            "preserve1": "modern digital devices",
            "preserve2": "minimalist aesthetic",
            "add": "golden hour lighting through windows",
            "text": "Transform to Art Deco 1920s luxury train car interior, preserve all modern digital devices, maintain overall minimalist aesthetic, AND add warm golden hour lighting through moving landscape windows"
        },
        {
            "transform": "medieval stone tower library",
            "preserve1": "contemporary ergonomic chairs",
            "preserve2": "cool LED task lighting",
            "add": "floating magical books and particles",
            "text": "Transform to medieval stone tower library with vaulted ceilings, preserve contemporary ergonomic office chairs, keep cool LED task lighting, AND add floating magical books and glowing particles"
        },
    ]
    
    idx = start_id
    for i in range(count):
        scene = BASE_SCENES[i % len(BASE_SCENES)]
        pattern = multi_patterns[i % len(multi_patterns)]
        
        prompts.append({
            "id": idx,
            "name": f"Multi: {pattern['transform'].split()[0]} + 3 Constraints",
            "category": "level_3_multi_constraint",
            "complexity_level": 3,
            "text": scene["text"],
            "theme": scene["theme"],
            "model": "fast",
            "resolution": "1024x1024",
            "edit_info": {
                "text": pattern["text"],
                "transformations": [
                    {
                        "type": "theme_change",
                        "target": pattern["transform"],
                        "priority": 1
                    },
                    {
                        "type": "atmospheric_addition",
                        "target": pattern["add"],
                        "priority": 4
                    }
                ],
                "constraints": [
                    {
                        "type": "preservation",
                        "target": pattern["preserve1"],
                        "priority": 2
                    },
                    {
                        "type": "preservation",
                        "target": pattern["preserve2"],
                        "priority": 3
                    }
                ],
                "expected_actions": [
                    "architecture_style",
                    "preserve_attribute",
                    "preserve_attribute",
                    "atmospheric_effects"
                ],
                "complexity_tags": [
                    "theme_transformation",
                    "multiple_preservation",
                    "atmospheric_addition",
                    "compositional"
                ]
            },
            "_cycle": 0,
            "_original_id": idx
        })
        idx += 1
    
    return prompts


def generate_level4_prompts(start_id=361, count=40):
    """Generate Level 4: Conditional logic prompts"""
    prompts = []
    
    conditional_patterns = [
        {
            "condition": "IF there are windows visible",
            "then": "show heavy rain outside with wet glass and water droplets",
            "otherwise": "add indoor water fountain feature with flowing water",
            "text": "IF there are windows visible in the scene, THEN show heavy rain outside with wet glass and water droplets; OTHERWISE add indoor decorative water fountain with flowing water sounds"
        },
        {
            "condition": "IF outdoor area is present",
            "then": "apply dense fog effect outdoors only with reduced visibility",
            "otherwise": "maintain current indoor clear atmosphere",
            "text": "IF outdoor area is visible (through windows or open spaces), THEN apply dense fog effect outdoors only with greatly reduced visibility; OTHERWISE maintain current indoor clear atmosphere unchanged"
        },
        {
            "condition": "IF there are lamps or light fixtures",
            "then": "make it nighttime and turn all lamps on with warm glow",
            "otherwise": "add candles throughout for ambient lighting",
            "text": "IF there are lamps or light fixtures present, THEN make it nighttime and turn all lamps on with warm glowing light; OTHERWISE add decorative candles throughout room for ambient lighting"
        },
        {
            "condition": "IF there are people visible",
            "then": "add festive party decorations and colorful lighting",
            "otherwise": "keep minimalist aesthetic with subtle changes only",
            "text": "IF there are people visible in the scene, THEN add festive party decorations with colorful string lights and balloons; OTHERWISE keep minimalist aesthetic with only subtle atmospheric changes"
        },
        {
            "condition": "IF modern technology is present",
            "then": "keep it and transform rest to Victorian era for contrast",
            "otherwise": "make full authentic Victorian transformation",
            "text": "IF modern technology devices are present (computers, TVs, etc), THEN keep them unchanged and transform everything else to Victorian era for anachronistic contrast; OTHERWISE make fully authentic Victorian period transformation"
        },
    ]
    
    idx = start_id
    for i in range(count):
        scene = BASE_SCENES[i % len(BASE_SCENES)]
        pattern = conditional_patterns[i % len(conditional_patterns)]
        
        prompts.append({
            "id": idx,
            "name": f"Conditional: {pattern['condition'].split()[1:4]}",
            "category": "level_4_conditional",
            "complexity_level": 4,
            "text": scene["text"],
            "theme": scene["theme"],
            "model": "fast",
            "resolution": "1024x1024",
            "edit_info": {
                "text": pattern["text"],
                "conditional_logic": {
                    "condition": pattern["condition"],
                    "then_action": pattern["then"],
                    "else_action": pattern["otherwise"]
                },
                "expected_actions": ["conditional_transform"],
                "complexity_tags": [
                    "conditional_logic",
                    "if_then_else",
                    "context_aware",
                    "advanced_reasoning"
                ]
            },
            "_cycle": 0,
            "_original_id": idx
        })
        idx += 1
    
    return prompts


def main():
    """Generate all 400 prompts and save to JSON"""
    print("🎨 Generating 400 complex prompts for unified dataset...")
    print()
    
    # Generate all levels
    print("  Level 1: Generating 120 dual transformation prompts...")
    level1 = generate_level1_prompts(start_id=1, count=120)
    
    print("  Level 2: Generating 160 transform + constraint prompts...")
    level2 = generate_level2_prompts(start_id=121, count=160)
    
    print("  Level 3: Generating 80 multi-constraint prompts...")
    level3 = generate_level3_prompts(start_id=281, count=80)
    
    print("  Level 4: Generating 40 conditional logic prompts...")
    level4 = generate_level4_prompts(start_id=361, count=40)
    
    # Combine all prompts
    all_prompts = level1 + level2 + level3 + level4
    
    print(f"\n✅ Generated {len(all_prompts)} total prompts")
    print(f"   Level 1 (Dual): {len(level1)}")
    print(f"   Level 2 (Constraint): {len(level2)}")
    print(f"   Level 3 (Multi): {len(level3)}")
    print(f"   Level 4 (Conditional): {len(level4)}")
    
    # Create output structure
    output = {
        "version": "1.0-complex",
        "description": "Complex unified dataset prompts with multi-constraint transformations",
        "statistics": {
            "total_prompts": len(all_prompts),
            "level_1_dual": len(level1),
            "level_2_constraint": len(level2),
            "level_3_multi": len(level3),
            "level_4_conditional": len(level4),
            "expected_samples_at_40_cycles": len(all_prompts) * 40
        },
        "complexity_levels": {
            "1": "Dual transformations (2 changes, no constraints)",
            "2": "Single transformation + 1 constraint (preserve/exclude)",
            "3": "Multi-transformation + multiple constraints",
            "4": "Conditional logic (if-then-else reasoning)"
        },
        "prompts": all_prompts
    }
    
    # Save to file
    output_dir = Path(__file__).parent.parent / "config" / "complex_prompts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "prompts_unified_complex.json"
    
    print(f"\n💾 Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully saved {len(all_prompts)} prompts!")
    print(f"\n📊 Expected dataset size: {len(all_prompts) * 40} samples (at 40 cycles)")
    print(f"   This script generates 400 prompts for 16,000 samples")


if __name__ == "__main__":
    main()

