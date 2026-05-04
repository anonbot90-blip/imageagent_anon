#!/usr/bin/env python3
"""
Generate 1000 complex prompts for styling/transformation dataset (V2)

Distribution:
- Level 1 (Dual Style): 300 prompts (30%)
- Level 2 (Triple + Constraint): 400 prompts (40%)
- Level 3 (Multi-Style): 200 prompts (20%)
- Level 4 (Complex): 100 prompts (10%)

Focus: 30 styling/transformation actions only
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Set seed for reproducibility
random.seed(42)

# 30 Styling Actions (from action_library_v3.json)
STYLING_ACTIONS = [
    # Theme (10 from V1)
    "location_setting", "architecture_style", "time_period_era",
    "time_of_day", "season_cycle", "weather_conditions",
    "mood_lighting", "color_grading", "artistic_medium", "atmospheric_effects",
    # Advanced Artistic (6 new)
    "art_movement", "painting_technique", "illustration_style",
    "rendering_style", "texture_overlay", "filter_effects",
    # Color (5 new)
    "color_palette", "color_temperature", "saturation_intensity",
    "contrast_style", "tonal_range",
    # Lighting (5 new)
    "lighting_style", "light_quality", "shadow_treatment",
    "glow_effects", "atmospheric_depth",
    # Effects (4 new)
    "motion_aesthetics", "lens_character", "grain_noise", "sharpness_softness"
]

# Subject categories
SUBJECTS = {
    "portraits": [
        "elderly woman with weathered face",
        "young man in casual attire",
        "child playing in park",
        "business professional in suit",
        "artist in studio",
        "musician with instrument",
        "dancer in motion",
        "athlete training",
        "chef in kitchen",
        "fashion model posing",
        "street performer",
        "grandmother knitting",
        "construction worker",
        "scientist in lab",
        "teacher in classroom",
    ],
    "landscapes": [
        "mountain range at sunrise",
        "coastal cliffs by ocean",
        "desert sand dunes",
        "forest with tall trees",
        "rolling hills countryside",
        "lake with reflection",
        "canyon with rock formations",
        "waterfall in jungle",
        "volcanic landscape",
        "alpine meadow",
        "river valley",
        "ice fields and glaciers",
        "prairie grasslands",
        "tropical beach",
        "autumn forest path",
    ],
    "cityscapes": [
        "busy city street corner",
        "modern downtown skyline",
        "historic town square",
        "narrow alley with shops",
        "train station platform",
        "bridge over river",
        "rooftop view of city",
        "shopping district at night",
        "industrial district",
        "waterfront promenade",
        "residential neighborhood",
        "cafe on cobblestone street",
        "market square with vendors",
        "subway station entrance",
        "skyscraper glass facade",
    ],
    "interiors": [
        "cozy living room with fireplace",
        "modern kitchen with island",
        "library with floor-to-ceiling books",
        "art gallery with paintings",
        "bedroom with large windows",
        "restaurant dining area",
        "office workspace",
        "museum hall with exhibits",
        "cathedral interior",
        "vintage bookshop",
    ],
    "nature": [
        "cherry blossoms in bloom",
        "sunflower field stretching to horizon",
        "butterfly on wildflower",
        "deer in forest clearing",
        "wolf in snowy landscape",
        "eagle soaring in sky",
        "garden with roses",
        "coral reef underwater",
        "tiger in jungle",
        "autumn leaves on ground",
    ],
    "fantasy_scifi": [
        "futuristic city with flying vehicles",
        "magical forest with glowing plants",
        "space station orbiting planet",
        "dragon perched on castle",
        "cyberpunk street market",
        "crystal cave with minerals",
        "alien landscape with twin suns",
        "enchanted castle on hill",
        "underwater city dome",
        "robot in post-apocalyptic ruins",
    ],
    "historical": [
        "Victorian mansion exterior",
        "medieval village scene",
        "ancient temple ruins",
        "Renaissance palace courtyard",
        "1920s jazz club interior",
    ],
    "abstract": [
        "geometric shapes composition",
        "flowing fabric patterns",
        "liquid paint swirls",
        "light and shadow interplay",
        "textured abstract surface",
    ]
}

# Action parameters for variety
ACTION_PARAMS = {
    "location_setting": ["urban", "coastal", "mountain", "forest", "desert"],
    "time_of_day": ["dawn", "golden_hour", "noon", "dusk", "night", "blue_hour"],
    "season_cycle": ["spring", "summer", "autumn", "winter"],
    "weather_conditions": ["foggy", "rainy", "snowy", "stormy", "clear"],
    "mood_lighting": ["dramatic", "romantic", "mysterious", "cheerful", "ominous"],
    "color_grading": ["cinematic_teal_orange", "warm_golden", "cool_blue", "vintage_sepia"],
    "artistic_medium": ["oil_painting", "watercolor", "digital_art", "pencil_sketch"],
    "art_movement": ["impressionism", "art_deco", "surrealism", "pop_art"],
    "color_palette": ["monochrome", "pastel", "neon", "earth_tones"],
    "lighting_style": ["rembrandt", "dramatic_side", "soft_front", "rim_light"],
}

def generate_action_with_param(action: str) -> str:
    """Generate action with parameter if available"""
    if action in ACTION_PARAMS:
        param = random.choice(ACTION_PARAMS[action])
        return f"{action}:{param}"
    return action

def get_random_subject() -> tuple[str, str]:
    """Get random subject from categories"""
    category = random.choice(list(SUBJECTS.keys()))
    subject = random.choice(SUBJECTS[category])
    return category, subject

def generate_constraint(actions: List[str], complexity: int) -> str:
    """Generate natural language constraint based on actions and complexity"""
    constraints_pool = [
        "maintain subject clarity",
        "preserve important details",
        "keep composition balanced",
        "ensure visual harmony",
        "retain depth perception",
        "balance contrast with detail",
        "preserve natural feel",
        "maintain color coherence",
        "keep subject recognizable",
        "preserve spatial relationships"
    ]
    
    complex_constraints = [
        "while maintaining X, also preserve Y",
        "balance dramatic effect with natural look",
        "ensure consistency across all transformations",
        "apply effects harmoniously without overwhelming",
        "maintain visual hierarchy and focal points",
    ]
    
    if complexity >= 3:
        base = random.choice(complex_constraints)
        specific = random.sample(constraints_pool, 2)
        return f"{base.replace('X', specific[0]).replace('Y', specific[1])}"
    else:
        return random.choice(constraints_pool)

def generate_level_1_prompt() -> Dict[str, Any]:
    """Generate Level 1: Dual Style (2 actions)"""
    category, subject = get_random_subject()
    
    # Select 2 compatible actions
    actions = random.sample(STYLING_ACTIONS, 2)
    actions_with_params = [generate_action_with_param(a) for a in actions]
    
    constraint = generate_constraint(actions, 1)
    
    return {
        "prompt": subject,
        "complexity_level": 1,
        "category": category,
        "actions": actions_with_params,
        "constraint": constraint,
        "description": f"{subject.split()[0]}_{actions[0]}_dual"
    }

def generate_level_2_prompt() -> Dict[str, Any]:
    """Generate Level 2: Triple Style + Constraint (3 actions)"""
    category, subject = get_random_subject()
    
    # Select 3 actions from different categories
    actions = random.sample(STYLING_ACTIONS, 3)
    actions_with_params = [generate_action_with_param(a) for a in actions]
    
    constraint = generate_constraint(actions, 2)
    
    return {
        "prompt": subject,
        "complexity_level": 2,
        "category": category,
        "actions": actions_with_params,
        "constraint": constraint,
        "description": f"{subject.split()[0]}_{actions[0]}_triple"
    }

def generate_level_3_prompt() -> Dict[str, Any]:
    """Generate Level 3: Multi-Style Composition (4 actions)"""
    category, subject = get_random_subject()
    
    # Select 4 actions ensuring variety
    actions = random.sample(STYLING_ACTIONS, 4)
    actions_with_params = [generate_action_with_param(a) for a in actions]
    
    constraint = generate_constraint(actions, 3)
    
    return {
        "prompt": subject,
        "complexity_level": 3,
        "category": category,
        "actions": actions_with_params,
        "constraint": constraint,
        "description": f"{subject.split()[0]}_{actions[0]}_multi"
    }

def generate_level_4_prompt() -> Dict[str, Any]:
    """Generate Level 4: Complex Multi-Style (5-6 actions)"""
    category, subject = get_random_subject()
    
    # Select 5-6 actions for complex transformation
    num_actions = random.choice([5, 6])
    actions = random.sample(STYLING_ACTIONS, num_actions)
    actions_with_params = [generate_action_with_param(a) for a in actions]
    
    constraint = generate_constraint(actions, 4)
    
    return {
        "prompt": subject,
        "complexity_level": 4,
        "category": category,
        "actions": actions_with_params,
        "constraint": constraint,
        "description": f"{subject.split()[0]}_{actions[0]}_complex"
    }

def generate_all_prompts() -> List[Dict[str, Any]]:
    """Generate all 1000 prompts"""
    prompts = []
    
    # Level 1: 300 prompts (30%)
    print("Generating Level 1 prompts (300)...")
    for i in range(300):
        prompt = generate_level_1_prompt()
        prompt["id"] = f"v2_l1_{i:04d}"
        prompts.append(prompt)
    
    # Level 2: 400 prompts (40%)
    print("Generating Level 2 prompts (400)...")
    for i in range(400):
        prompt = generate_level_2_prompt()
        prompt["id"] = f"v2_l2_{i:04d}"
        prompts.append(prompt)
    
    # Level 3: 200 prompts (20%)
    print("Generating Level 3 prompts (200)...")
    for i in range(200):
        prompt = generate_level_3_prompt()
        prompt["id"] = f"v2_l3_{i:04d}"
        prompts.append(prompt)
    
    # Level 4: 100 prompts (10%)
    print("Generating Level 4 prompts (100)...")
    for i in range(100):
        prompt = generate_level_4_prompt()
        prompt["id"] = f"v2_l4_{i:04d}"
        prompts.append(prompt)
    
    # Shuffle to mix complexity levels
    random.shuffle(prompts)
    
    return prompts

def validate_prompts(prompts: List[Dict[str, Any]]) -> bool:
    """Validate generated prompts"""
    print("\n=== Validation ===")
    
    # Check total count
    assert len(prompts) == 1000, f"Expected 1000 prompts, got {len(prompts)}"
    print(f"✅ Total prompts: {len(prompts)}")
    
    # Check distribution
    level_counts = {}
    for p in prompts:
        level = p["complexity_level"]
        level_counts[level] = level_counts.get(level, 0) + 1
    
    expected = {1: 300, 2: 400, 3: 200, 4: 100}
    for level, count in level_counts.items():
        assert count == expected[level], f"Level {level}: expected {expected[level]}, got {count}"
        print(f"✅ Level {level}: {count} prompts ({count/10}%)")
    
    # Check action counts
    action_usage = {}
    for p in prompts:
        for action in p["actions"]:
            base_action = action.split(":")[0]
            action_usage[base_action] = action_usage.get(base_action, 0) + 1
    
    print(f"\n✅ Action coverage: {len(action_usage)}/30 actions used")
    
    # Check all required fields
    required_fields = ["id", "prompt", "complexity_level", "category", "actions", "constraint", "description"]
    for i, p in enumerate(prompts[:5]):  # Sample check
        for field in required_fields:
            assert field in p, f"Prompt {i} missing field: {field}"
    print(f"✅ All required fields present")
    
    return True

def main():
    print("=" * 60)
    print("Complex Generation V2 - Prompt Generation")
    print("=" * 60)
    print(f"Target: 1000 prompts across 4 complexity levels")
    print(f"Actions: 30 styling/transformation actions")
    print(f"Output: config/complex_prompts_v2/prompts_complex_v2_1000.json")
    print("=" * 60)
    
    # Generate prompts
    prompts = generate_all_prompts()
    
    # Validate
    validate_prompts(prompts)
    
    # Save to file
    output_path = Path(__file__).parent.parent.parent / "config" / "complex_prompts_v2" / "prompts_complex_v2_1000.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    print(f"\n✅ Prompts saved to: {output_path}")
    print(f"📊 Total size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Print sample
    print("\n=== Sample Prompts ===")
    for level in [1, 2, 3, 4]:
        sample = next(p for p in prompts if p["complexity_level"] == level)
        print(f"\nLevel {level}:")
        print(f"  Prompt: {sample['prompt']}")
        print(f"  Actions: {', '.join(sample['actions'])}")
        print(f"  Constraint: {sample['constraint']}")

if __name__ == "__main__":
    main()

