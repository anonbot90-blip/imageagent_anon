#!/usr/bin/env python3
"""
Expand prompt_theme_100.json from 100 to 500 prompts
Generates 400 new diverse transformation prompts (IDs 101-500)
"""

import json
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

def generate_new_prompts():
    """Generate 400 new diverse prompts covering all transformation types"""
    
    new_prompts = []
    prompt_id = 101
    
    # ========================================================================
    # LOCATION TRANSFORMATIONS (101-170) - 70 prompts
    # ========================================================================
    location_transforms = [
        # Natural environments
        ("Forest to Desert", "dense temperate forest with tall trees, lush undergrowth, dappled sunlight, and rich green atmosphere", "forest_temperate", 
         "vast sandy desert with rolling dunes, sparse cacti, intense sunlight, endless horizon, and arid atmosphere", "desert_sand", "forest_to_desert"),
        
        ("Underwater Reef to Sky Islands", "vibrant underwater coral reef with colorful fish, swaying seaweed, filtered blue light, and aquatic atmosphere", "underwater_reef",
         "floating sky islands with waterfalls cascading into clouds, flying creatures, ethereal mist, and dreamlike atmosphere", "sky_floating", "underwater_to_sky"),
        
        ("Arctic Tundra to Volcanic Wasteland", "frozen arctic tundra with ice sheets, aurora borealis, polar wildlife, and frigid atmosphere", "arctic_ice",
         "active volcanic landscape with flowing lava, ash clouds, glowing magma, and intense heat atmosphere", "volcano_lava", "arctic_to_volcano"),
        
        ("Dark Cave to Grand Palace", "mysterious cave interior with stalactites, underground pools, dim natural light, and ancient atmosphere", "cave_underground",
         "opulent palace hall with marble columns, crystal chandeliers, ornate decorations, and regal atmosphere", "palace_interior", "cave_to_palace"),
        
        ("Misty Swamp to Mountain Peak", "murky swamp with twisted trees, hanging moss, still water, fog, and mysterious atmosphere", "swamp_wetland",
         "majestic mountain peak with snow cap, rocky cliffs, thin air, panoramic views, and elevated atmosphere", "mountain_rocky", "swamp_to_mountain"),
        
        ("Industrial Warehouse to Tropical Jungle", "industrial warehouse with metal beams, concrete floors, machinery, and utilitarian atmosphere", "industrial_zone",
         "lush tropical jungle with dense canopy, exotic plants, wildlife sounds, humidity, and vibrant atmosphere", "jungle_tropical", "industrial_to_tropical"),
        
        ("Rural Farmland to Space Station", "peaceful rural farmland with crops, barn, windmill, open fields, and pastoral atmosphere", "rural_village",
         "advanced space station interior with zero gravity, control panels, viewing windows to stars, and futuristic atmosphere", "space_station", "rural_to_space"),
        
        ("Ancient Temple to Cyberpunk Alley", "ancient stone temple with carved pillars, incense smoke, prayer offerings, and sacred atmosphere", "temple_shrine",
         "neon-lit cyberpunk alley with holographic signs, steam vents, tech vendors, and gritty futuristic atmosphere", "cyberpunk_city", "temple_to_cyberpunk"),
        
        ("Cozy Bedroom to Alien Planet", "comfortable bedroom with soft bed, warm lighting, personal items, and intimate atmosphere", "bedroom_interior",
         "alien planet surface with strange rock formations, multiple moons, exotic flora, and otherworldly atmosphere", "alien_planet", "bedroom_to_alien"),
        
        ("Deep Canyon to Open Ocean", "dramatic canyon with layered rock walls, winding river below, desert plants, and geological atmosphere", "canyon_valley",
         "vast open ocean with rolling waves, distant horizon, seabirds, and maritime atmosphere", "ocean_open_water", "canyon_to_ocean"),
        
        # Urban/Built environments
        ("Shopping Mall to Haunted Mansion", "modern shopping mall with bright stores, escalators, shoppers, and commercial atmosphere", "commercial_district",
         "abandoned haunted mansion with dusty furniture, broken windows, ghostly presence, and eerie atmosphere", "haunted_mansion", "mall_to_haunted"),
        
        ("Suburban Home to Post-Apocalyptic Ruins", "neat suburban home with manicured lawn, white picket fence, and peaceful atmosphere", "suburban_neighborhood",
         "post-apocalyptic ruins with crumbling buildings, overgrown vegetation, abandoned vehicles, and desolate atmosphere", "post_apocalyptic", "suburban_to_ruins"),
        
        ("City Park to Medieval Castle", "urban park with walking paths, playground, flower beds, and recreational atmosphere", "urban_city",
         "medieval castle courtyard with stone walls, towers, banners, guards, and historical atmosphere", "fantasy_castle", "park_to_castle"),
        
        ("Office Building to Tropical Island", "corporate office building with cubicles, conference rooms, fluorescent lights, and professional atmosphere", "office_modern",
         "pristine tropical island with white beaches, palm trees, clear water, and paradise atmosphere", "tropical_island", "office_to_island"),
        
        ("Library to Alien Spacecraft", "quiet library with bookshelves, reading tables, soft lighting, and scholarly atmosphere", "office_modern",
         "alien spacecraft interior with bioluminescent panels, organic architecture, strange technology, and extraterrestrial atmosphere", "space_station", "library_to_spacecraft"),
        
        ("Restaurant to Medieval Tavern", "modern restaurant with elegant tables, ambient lighting, contemporary decor, and dining atmosphere", "commercial_district",
         "rustic medieval tavern with wooden benches, stone fireplace, ale barrels, and historical atmosphere", "medieval_town", "restaurant_to_tavern"),
        
        ("Gym to Gladiator Arena", "modern fitness gym with equipment, mirrors, weights, and athletic atmosphere", "commercial_district",
         "ancient Roman gladiator arena with sand floor, spectator stands, weapons racks, and combat atmosphere", "fantasy_castle", "gym_to_arena"),
        
        ("Hospital to Steampunk Laboratory", "clean hospital corridor with medical equipment, white walls, and clinical atmosphere", "office_modern",
         "steampunk laboratory with brass machinery, steam pipes, gears, and Victorian-tech atmosphere", "industrial_zone", "hospital_to_steampunk"),
        
        ("School Classroom to Wizard Academy", "modern classroom with desks, whiteboard, computers, and educational atmosphere", "office_modern",
         "magical wizard academy with floating books, crystal balls, spell ingredients, and mystical atmosphere", "fantasy_castle", "classroom_to_wizard"),
        
        ("Train Station to Wild West Town", "busy train station with platforms, schedules, commuters, and transit atmosphere", "urban_city",
         "dusty Wild West town with saloon, hitching posts, tumbleweeds, and frontier atmosphere", "rural_village", "station_to_wildwest"),
    ]
    
    for name, gen_text, gen_theme, edit_text, edit_theme, trans_type in location_transforms[:20]:  # First 20
        new_prompts.append({
            "id": prompt_id,
            "name": name,
            "category": "location_transformation",
            "generation": {
                "text": f"A {gen_text}",
                "theme": gen_theme,
                "model": "fast",
                "resolution": "1024x1024"
            },
            "edit": {
                "text": f"Transform to {edit_text}",
                "target_theme": edit_theme,
                "transformation_type": trans_type
            }
        })
        prompt_id += 1
    
    # Continue with more location transformations to reach 70 total
    # Adding 50 more diverse location pairs
    additional_locations = [
        ("Beach to Snowy Forest", "sunny beach", "beach_coast", "snowy forest", "forest_temperate"),
        ("Grassland to Urban City", "open grassland", "plains_grassland", "bustling urban city", "urban_city"),
        ("Mountain Cave to Underwater City", "mountain cave", "cave_underground", "underwater city", "underwater_reef"),
        ("Desert Oasis to Arctic Base", "desert oasis", "desert_sand", "arctic research base", "arctic_ice"),
        ("Jungle Temple to Modern Museum", "jungle temple ruins", "jungle_tropical", "modern art museum", "office_modern"),
        ("Volcano Crater to Ice Palace", "volcano crater", "volcano_lava", "ice palace", "arctic_ice"),
        ("Coral Reef to Space Colony", "coral reef", "underwater_reef", "space colony", "space_station"),
        ("Bamboo Forest to Neon City", "bamboo forest", "forest_temperate", "neon city", "cyberpunk_city"),
        ("Rocky Canyon to Cloud Kingdom", "rocky canyon", "canyon_valley", "cloud kingdom", "sky_floating"),
        ("Mangrove Swamp to Crystal Cave", "mangrove swamp", "swamp_wetland", "crystal cave", "cave_underground"),
        ("Savanna to Frozen Lake", "African savanna", "plains_grassland", "frozen lake", "arctic_ice"),
        ("Rainforest to Martian Surface", "rainforest", "jungle_tropical", "Martian surface", "alien_planet"),
        ("Wheat Field to Lava Flow", "wheat field", "plains_grassland", "lava flow", "volcano_lava"),
        ("Pine Forest to Coral Garden", "pine forest", "forest_temperate", "coral garden", "underwater_reef"),
        ("Sand Dunes to Moss Garden", "sand dunes", "desert_sand", "moss garden", "forest_temperate"),
        ("Cliff Edge to Deep Sea Trench", "cliff edge", "mountain_rocky", "deep sea trench", "underwater_reef"),
        ("Meadow to Asteroid Field", "flower meadow", "plains_grassland", "asteroid field", "space_station"),
        ("Wetland to Lava Cave", "wetland", "swamp_wetland", "lava cave", "volcano_lava"),
        ("Tundra to Tropical Beach", "tundra", "arctic_ice", "tropical beach", "tropical_island"),
        ("Forest Clearing to Underwater Ruins", "forest clearing", "forest_temperate", "underwater ruins", "underwater_reef"),
        ("Desert Temple to Ice Cavern", "desert temple", "desert_sand", "ice cavern", "arctic_ice"),
        ("Jungle Waterfall to Space Nebula", "jungle waterfall", "jungle_tropical", "space nebula", "space_station"),
        ("Mountain Pass to Ocean Floor", "mountain pass", "mountain_rocky", "ocean floor", "underwater_reef"),
        ("Volcanic Island to Frozen Wasteland", "volcanic island", "volcano_lava", "frozen wasteland", "arctic_ice"),
        ("Reef Lagoon to Desert Mirage", "reef lagoon", "underwater_reef", "desert mirage", "desert_sand"),
        ("Ice Sheet to Jungle Canopy", "ice sheet", "arctic_ice", "jungle canopy", "jungle_tropical"),
        ("Canyon River to Sky Platform", "canyon river", "canyon_valley", "sky platform", "sky_floating"),
        ("Swamp Village to Mountain Temple", "swamp village", "swamp_wetland", "mountain temple", "mountain_rocky"),
        ("Forest Lake to Lava Pool", "forest lake", "forest_temperate", "lava pool", "volcano_lava"),
        ("Desert Fort to Underwater Dome", "desert fort", "desert_sand", "underwater dome", "underwater_reef"),
        ("Arctic Harbor to Tropical Port", "arctic harbor", "arctic_ice", "tropical port", "tropical_island"),
        ("Jungle Bridge to Space Bridge", "jungle bridge", "jungle_tropical", "space bridge", "space_station"),
        ("Mountain Lodge to Beach Hut", "mountain lodge", "mountain_rocky", "beach hut", "beach_coast"),
        ("Volcanic Vent to Ice Geyser", "volcanic vent", "volcano_lava", "ice geyser", "arctic_ice"),
        ("Coral Atoll to Desert Oasis", "coral atoll", "underwater_reef", "desert oasis", "desert_sand"),
        ("Tundra Camp to Jungle Camp", "tundra camp", "arctic_ice", "jungle camp", "jungle_tropical"),
        ("Canyon Bridge to Ocean Bridge", "canyon bridge", "canyon_valley", "ocean bridge", "ocean_open_water"),
        ("Swamp Dock to Mountain Dock", "swamp dock", "swamp_wetland", "mountain dock", "mountain_rocky"),
        ("Forest Path to Desert Trail", "forest path", "forest_temperate", "desert trail", "desert_sand"),
        ("Ice Cave to Lava Tube", "ice cave", "arctic_ice", "lava tube", "volcano_lava"),
        ("Reef Channel to Space Corridor", "reef channel", "underwater_reef", "space corridor", "space_station"),
        ("Jungle Shrine to Arctic Shrine", "jungle shrine", "jungle_tropical", "arctic shrine", "arctic_ice"),
        ("Mountain Peak to Ocean Depth", "mountain peak", "mountain_rocky", "ocean depth", "underwater_reef"),
        ("Volcanic Beach to Icy Beach", "volcanic beach", "volcano_lava", "icy beach", "arctic_ice"),
        ("Desert Valley to Forest Valley", "desert valley", "desert_sand", "forest valley", "forest_temperate"),
        ("Swamp Tree to Mountain Tree", "swamp tree", "swamp_wetland", "mountain tree", "mountain_rocky"),
        ("Jungle River to Arctic River", "jungle river", "jungle_tropical", "arctic river", "arctic_ice"),
        ("Canyon Pool to Ocean Pool", "canyon pool", "canyon_valley", "ocean pool", "ocean_open_water"),
        ("Forest Ruins to Desert Ruins", "forest ruins", "forest_temperate", "desert ruins", "desert_sand"),
        ("Ice Fortress to Lava Fortress", "ice fortress", "arctic_ice", "lava fortress", "volcano_lava"),
    ]
    
    for name, gen_desc, gen_theme, edit_desc, edit_theme in additional_locations:
        new_prompts.append({
            "id": prompt_id,
            "name": name,
            "category": "location_transformation",
            "generation": {
                "text": f"A {gen_desc} with natural features, environmental details, and characteristic atmosphere",
                "theme": gen_theme,
                "model": "fast",
                "resolution": "1024x1024"
            },
            "edit": {
                "text": f"Transform to {edit_desc} with distinctive features, environmental characteristics, and unique atmosphere",
                "target_theme": edit_theme,
                "transformation_type": f"{gen_theme}_to_{edit_theme}"
            }
        })
        prompt_id += 1
    
    print(f"Generated location transformations: {prompt_id - 101} prompts (IDs 101-{prompt_id-1})")
    
    # ========================================================================
    # TIME/ERA TRANSFORMATIONS (171-240) - 70 prompts
    # ========================================================================
    
    era_transforms = [
        ("Ancient Egypt to Modern Times", "ancient Egyptian marketplace", "ancient_classical", "modern shopping district", "contemporary_2000s"),
        ("Medieval Castle to Future Fortress", "medieval stone castle", "medieval_dark_ages", "futuristic fortress", "far_future_2200"),
        ("Victorian Street to Cyberpunk Street", "Victorian era street", "victorian_1800s", "cyberpunk street", "cyberpunk_dystopia"),
        ("Stone Age Cave to Smart Home", "prehistoric cave dwelling", "prehistoric_stone_age", "smart home interior", "near_future_2050"),
        ("Renaissance Court to Space Station", "Renaissance royal court", "renaissance_1500s", "space station command", "far_future_2200"),
        ("Wild West Saloon to Neon Bar", "Wild West saloon", "victorian_1800s", "neon-lit futuristic bar", "cyberpunk_dystopia"),
        ("Roman Forum to Modern Plaza", "ancient Roman forum", "ancient_classical", "modern city plaza", "contemporary_2000s"),
        ("Art Deco Ballroom to Holographic Club", "1920s Art Deco ballroom", "art_deco_1920s", "holographic nightclub", "far_future_2200"),
        ("Baroque Palace to Minimalist Loft", "ornate Baroque palace", "baroque_1700s", "minimalist modern loft", "contemporary_2000s"),
        ("Medieval Market to Digital Marketplace", "medieval market square", "medieval_dark_ages", "virtual reality marketplace", "near_future_2050"),
        ("Ancient Library to Digital Archive", "ancient scroll library", "ancient_classical", "holographic archive", "far_future_2200"),
        ("Victorian Factory to Automated Plant", "Victorian steam factory", "victorian_1800s", "fully automated factory", "near_future_2050"),
        ("Renaissance Workshop to 3D Print Lab", "Renaissance artist workshop", "renaissance_1500s", "3D printing laboratory", "contemporary_2000s"),
        ("Stone Age Hunting to Drone Farming", "stone age hunting ground", "prehistoric_stone_age", "drone farming operation", "near_future_2050"),
        ("Roman Bath to Spa Pod", "Roman bathhouse", "ancient_classical", "futuristic spa pod", "far_future_2200"),
        ("Medieval Forge to Nanotech Lab", "medieval blacksmith forge", "medieval_dark_ages", "nanotechnology lab", "far_future_2200"),
        ("Art Nouveau Cafe to Hologram Lounge", "Art Nouveau cafe", "art_nouveau_1900s", "hologram lounge", "cyberpunk_dystopia"),
        ("Baroque Theater to VR Theater", "Baroque opera theater", "baroque_1700s", "virtual reality theater", "near_future_2050"),
        ("Victorian Study to AI Office", "Victorian gentleman's study", "victorian_1800s", "AI-powered office", "near_future_2050"),
        ("Ancient Temple to Energy Shrine", "ancient temple", "ancient_classical", "energy shrine", "far_future_2200"),
        ("Medieval Kitchen to Molecular Gastronomy", "medieval castle kitchen", "medieval_dark_ages", "molecular gastronomy lab", "contemporary_2000s"),
        ("Renaissance Garden to Vertical Farm", "Renaissance garden", "renaissance_1500s", "vertical hydroponic farm", "near_future_2050"),
        ("Stone Age Camp to Mars Colony", "stone age campsite", "prehistoric_stone_age", "Mars colony habitat", "far_future_2200"),
        ("Roman Arena to Hologram Stadium", "Roman colosseum", "ancient_classical", "holographic sports stadium", "far_future_2200"),
        ("Victorian Train to Hyperloop", "Victorian steam train", "victorian_1800s", "hyperloop transport", "near_future_2050"),
        ("Art Deco Hotel to Space Hotel", "Art Deco hotel lobby", "art_deco_1920s", "orbital space hotel", "far_future_2200"),
        ("Medieval Cathedral to Light Temple", "Gothic cathedral", "medieval_dark_ages", "temple of light", "far_future_2200"),
        ("Ancient Agora to Virtual Forum", "ancient Greek agora", "ancient_classical", "virtual reality forum", "near_future_2050"),
        ("Victorian Hospital to Medical Bay", "Victorian hospital ward", "victorian_1800s", "starship medical bay", "far_future_2200"),
        ("Renaissance Studio to Hologram Studio", "Renaissance art studio", "renaissance_1500s", "holographic art studio", "near_future_2050"),
        ("Stone Age Shelter to Biodome", "stone age shelter", "prehistoric_stone_age", "biodome habitat", "near_future_2050"),
        ("Roman Villa to Smart Villa", "Roman villa", "ancient_classical", "AI smart villa", "near_future_2050"),
        ("Medieval Tower to Comm Tower", "medieval watch tower", "medieval_dark_ages", "communications tower", "contemporary_2000s"),
        ("Art Nouveau Gallery to Digital Gallery", "Art Nouveau gallery", "art_nouveau_1900s", "digital art gallery", "contemporary_2000s"),
        ("Baroque Chapel to Meditation Pod", "Baroque chapel", "baroque_1700s", "meditation pod", "near_future_2050"),
        ("Victorian Pharmacy to Nanotech Clinic", "Victorian pharmacy", "victorian_1800s", "nanotechnology clinic", "far_future_2200"),
        ("Ancient Observatory to Space Telescope", "ancient observatory", "ancient_classical", "space telescope station", "far_future_2200"),
        ("Medieval Armory to Weapon Lab", "medieval armory", "medieval_dark_ages", "advanced weapons lab", "near_future_2050"),
        ("Renaissance Bank to Crypto Exchange", "Renaissance bank", "renaissance_1500s", "cryptocurrency exchange", "contemporary_2000s"),
        ("Stone Age Tool Shop to Fabricator", "stone age tool making", "prehistoric_stone_age", "matter fabricator", "far_future_2200"),
        ("Roman Port to Spaceport", "Roman seaport", "ancient_classical", "orbital spaceport", "far_future_2200"),
        ("Victorian Greenhouse to Biodome", "Victorian greenhouse", "victorian_1800s", "climate-controlled biodome", "near_future_2050"),
        ("Art Deco Cinema to Immersive Theater", "Art Deco cinema", "art_deco_1920s", "immersive experience theater", "near_future_2050"),
        ("Medieval Monastery to Research Station", "medieval monastery", "medieval_dark_ages", "research station", "contemporary_2000s"),
        ("Ancient Marketplace to Trade Hub", "ancient marketplace", "ancient_classical", "interstellar trade hub", "far_future_2200"),
        ("Victorian Mansion to Smart Mansion", "Victorian mansion", "victorian_1800s", "AI smart mansion", "near_future_2050"),
        ("Renaissance Square to Hologram Plaza", "Renaissance town square", "renaissance_1500s", "holographic plaza", "far_future_2200"),
        ("Stone Age Village to Orbital Colony", "stone age village", "prehistoric_stone_age", "orbital colony", "far_future_2200"),
        ("Roman Road to Hyperway", "Roman road", "ancient_classical", "hyperway transport", "near_future_2050"),
        ("Medieval Bridge to Energy Bridge", "medieval stone bridge", "medieval_dark_ages", "energy bridge", "far_future_2200"),
        ("Art Nouveau Station to Teleport Hub", "Art Nouveau train station", "art_nouveau_1900s", "teleportation hub", "far_future_2200"),
        ("Baroque Garden to Zen Garden", "Baroque formal garden", "baroque_1700s", "minimalist zen garden", "contemporary_2000s"),
        ("Victorian Workshop to Maker Space", "Victorian workshop", "victorian_1800s", "modern maker space", "contemporary_2000s"),
        ("Ancient School to Learning Pod", "ancient school", "ancient_classical", "AI learning pod", "near_future_2050"),
        ("Medieval Inn to Capsule Hotel", "medieval inn", "medieval_dark_ages", "capsule hotel", "contemporary_2000s"),
        ("Renaissance Theater to Projection Dome", "Renaissance theater", "renaissance_1500s", "360 projection dome", "near_future_2050"),
        ("Stone Age Fire to Fusion Reactor", "stone age fire pit", "prehistoric_stone_age", "fusion reactor", "far_future_2200"),
        ("Roman Aqueduct to Water Processor", "Roman aqueduct", "ancient_classical", "water processing facility", "near_future_2050"),
        ("Victorian Parlor to Lounge Pod", "Victorian parlor", "victorian_1800s", "relaxation pod", "near_future_2050"),
        ("Art Deco Office to Cloud Office", "Art Deco office", "art_deco_1920s", "cloud-based office", "contemporary_2000s"),
        ("Medieval Scriptorium to Data Center", "medieval scriptorium", "medieval_dark_ages", "data center", "contemporary_2000s"),
        ("Ancient Gymnasium to Training Sim", "ancient gymnasium", "ancient_classical", "virtual training simulator", "near_future_2050"),
        ("Victorian Conservatory to Climate Dome", "Victorian conservatory", "victorian_1800s", "climate control dome", "far_future_2200"),
        ("Renaissance Fountain to Hologram Display", "Renaissance fountain", "renaissance_1500s", "holographic display fountain", "near_future_2050"),
        ("Stone Age Burial to Memory Archive", "stone age burial site", "prehistoric_stone_age", "digital memory archive", "far_future_2200"),
        ("Roman Temple to Quantum Shrine", "Roman temple", "ancient_classical", "quantum energy shrine", "far_future_2200"),
        ("Medieval Dungeon to Detention Field", "medieval dungeon", "medieval_dark_ages", "force field detention", "far_future_2200"),
        ("Art Nouveau Bridge to Light Bridge", "Art Nouveau bridge", "art_nouveau_1900s", "hard light bridge", "far_future_2200"),
        ("Baroque Fountain to Energy Fountain", "Baroque fountain", "baroque_1700s", "energy fountain", "far_future_2200"),
        ("Victorian Clock Tower to Time Station", "Victorian clock tower", "victorian_1800s", "time monitoring station", "far_future_2200"),
    ]
    
    for name, gen_desc, gen_era, edit_desc, edit_era in era_transforms:
        new_prompts.append({
            "id": prompt_id,
            "name": name,
            "category": "era_transformation",
            "generation": {
                "text": f"A {gen_desc} with period-appropriate details, historical architecture, and era-specific atmosphere",
                "theme": gen_era,
                "model": "fast",
                "resolution": "1024x1024"
            },
            "edit": {
                "text": f"Transform to {edit_desc} with futuristic technology, advanced design, and era-appropriate atmosphere",
                "target_theme": edit_era,
                "transformation_type": f"{gen_era}_to_{edit_era}"
            }
        })
        prompt_id += 1
    
    print(f"Generated era transformations: {len(era_transforms)} prompts (IDs 171-{prompt_id-1})")
    
    # ========================================================================
    # SEASONAL/WEATHER TRANSFORMATIONS (241-310) - 70 prompts
    # ========================================================================
    
    seasonal_weather = [
        ("Spring Meadow to Autumn Field", "spring meadow", "spring_blooming", "autumn field", "autumn_falling"),
        ("Summer Beach to Winter Beach", "summer beach", "summer_lush", "winter beach", "winter_bare"),
        ("Autumn Forest to Spring Forest", "autumn forest", "autumn_falling", "spring forest", "spring_blooming"),
        ("Winter Mountain to Summer Mountain", "winter mountain", "winter_bare", "summer mountain", "summer_lush"),
        ("Clear Sky to Thunderstorm", "clear sunny sky", "clear_sunny", "dramatic thunderstorm", "thunderstorm"),
        ("Foggy Morning to Clear Afternoon", "foggy morning", "fog_dense", "clear afternoon", "clear_sunny"),
        ("Rainy Day to Snowy Day", "rainy day", "heavy_rain", "snowy day", "heavy_snow_blizzard"),
        ("Sunny Garden to Stormy Garden", "sunny garden", "clear_sunny", "stormy garden", "thunderstorm"),
        ("Misty Lake to Frozen Lake", "misty lake", "mist_light", "frozen lake", "heavy_snow_blizzard"),
        ("Dry Desert to Sandstorm", "dry desert", "clear_sunny", "sandstorm desert", "sandstorm"),
    ]
    
    # Generate 70 seasonal/weather transformations
    seasons = ["spring", "summer", "autumn", "winter"]
    weather_types = ["clear", "rainy", "snowy", "foggy", "stormy", "misty", "overcast"]
    locations_for_weather = ["park", "street", "garden", "forest", "beach", "mountain", "field", "lake", "river", "valley"]
    
    for i in range(70):
        if i < len(seasonal_weather):
            name, gen_desc, gen_season, edit_desc, edit_season = seasonal_weather[i]
        else:
            # Generate more combinations
            loc = locations_for_weather[i % len(locations_for_weather)]
            season1 = seasons[i % len(seasons)]
            season2 = seasons[(i + 2) % len(seasons)]
            name = f"{season1.capitalize()} {loc.capitalize()} to {season2.capitalize()} {loc.capitalize()}"
            gen_desc = f"{season1} {loc}"
            gen_season = f"{season1}_blooming" if season1 == "spring" else f"{season1}_lush" if season1 == "summer" else f"{season1}_falling" if season1 == "autumn" else "winter_bare"
            edit_desc = f"{season2} {loc}"
            edit_season = f"{season2}_blooming" if season2 == "spring" else f"{season2}_lush" if season2 == "summer" else f"{season2}_falling" if season2 == "autumn" else "winter_bare"
        
        new_prompts.append({
            "id": prompt_id,
            "name": name,
            "category": "seasonal_transformation",
            "generation": {
                "text": f"A {gen_desc} with seasonal characteristics, weather conditions, and atmospheric effects",
                "theme": gen_season,
                "model": "fast",
                "resolution": "1024x1024"
            },
            "edit": {
                "text": f"Transform to {edit_desc} with different seasonal features, weather patterns, and atmospheric changes",
                "target_theme": edit_season,
                "transformation_type": f"{gen_season}_to_{edit_season}"
            }
        })
        prompt_id += 1
    
    print(f"Generated seasonal/weather transformations: 70 prompts (IDs 241-{prompt_id-1})")
    
    # ========================================================================
    # MOOD/ATMOSPHERE TRANSFORMATIONS (311-380) - 70 prompts
    # ========================================================================
    
    mood_transforms = [
        ("Cheerful Cafe to Noir Cafe", "bright cheerful", "mysterious_shadowy"),
        ("Peaceful Garden to Tense Garden", "serene_calm", "tense_suspenseful"),
        ("Romantic Balcony to Ominous Balcony", "romantic_intimate", "ominous_dark"),
        ("Whimsical Playground to Eerie Playground", "whimsical_playful", "eerie_unsettling"),
        ("Energetic Street to Melancholic Street", "energetic_vibrant", "melancholic_somber"),
        ("Hopeful Sunrise to Dramatic Sunset", "hopeful_uplifting", "dramatic_cinematic"),
        ("Cozy Library to Mysterious Library", "soft_cozy", "mysterious_shadowy"),
        ("Epic Battlefield to Serene Meadow", "epic_grand", "serene_calm"),
        ("Bright Classroom to Dark Classroom", "bright_cheerful", "ominous_dark"),
        ("Nostalgic Street to Modern Street", "nostalgic_sepia", "bright_cheerful"),
    ]
    
    moods = ["bright_cheerful", "soft_cozy", "mysterious_shadowy", "dramatic_cinematic", "ominous_dark",
             "romantic_intimate", "energetic_vibrant", "melancholic_somber", "whimsical_playful",
             "nostalgic_sepia", "tense_suspenseful", "serene_calm", "eerie_unsettling", "epic_grand", "hopeful_uplifting"]
    
    locations_for_mood = ["room", "hallway", "street", "park", "building", "bridge", "plaza", "alley", "courtyard", "corridor"]
    
    for i in range(70):
        if i < len(mood_transforms):
            name, mood1, mood2 = mood_transforms[i]
            loc = name.split()[1]
        else:
            loc = locations_for_mood[i % len(locations_for_mood)]
            mood1 = moods[i % len(moods)]
            mood2 = moods[(i + 7) % len(moods)]
            mood1_name = mood1.replace("_", " ").title()
            mood2_name = mood2.replace("_", " ").title()
            name = f"{mood1_name} {loc.capitalize()} to {mood2_name} {loc.capitalize()}"
        
        new_prompts.append({
            "id": prompt_id,
            "name": name,
            "category": "mood_transformation",
            "generation": {
                "text": f"A {loc} with {mood1.replace('_', ' ')} mood, appropriate lighting, and emotional atmosphere",
                "theme": mood1,
                "model": "fast",
                "resolution": "1024x1024"
            },
            "edit": {
                "text": f"Transform to {mood2.replace('_', ' ')} mood with different lighting, altered atmosphere, and new emotional tone",
                "target_theme": mood2,
                "transformation_type": f"{mood1}_to_{mood2}"
            }
        })
        prompt_id += 1
    
    print(f"Generated mood transformations: 70 prompts (IDs 311-{prompt_id-1})")
    
    # ========================================================================
    # ARTISTIC STYLE TRANSFORMATIONS (381-450) - 70 prompts
    # ========================================================================
    
    artistic_styles = [
        ("photorealistic", "oil_painting_thick"),
        ("photorealistic", "watercolor_flowing"),
        ("photorealistic", "anime_cel_shaded"),
        ("photorealistic", "pixel_art"),
        ("photorealistic", "pencil_sketch"),
        ("oil_painting_thick", "digital_painting"),
        ("watercolor_flowing", "acrylic_painting"),
        ("anime_cel_shaded", "cartoon_illustrated"),
        ("pixel_art", "low_poly"),
        ("pencil_sketch", "charcoal_sketch"),
    ]
    
    all_styles = ["photorealistic", "hyperrealistic", "oil_painting_thick", "oil_painting_smooth",
                  "watercolor_flowing", "watercolor_dry", "acrylic_painting", "ink_brush",
                  "pencil_sketch", "charcoal_sketch", "digital_painting", "vector_art",
                  "3d_rendered", "low_poly", "pixel_art", "anime_cel_shaded", "cartoon_illustrated"]
    
    subjects = ["portrait", "landscape", "cityscape", "still life", "architecture", "nature scene", "interior"]
    
    for i in range(70):
        if i < len(artistic_styles):
            style1, style2 = artistic_styles[i]
        else:
            style1 = all_styles[i % len(all_styles)]
            style2 = all_styles[(i + 5) % len(all_styles)]
        
        subject = subjects[i % len(subjects)]
        style1_name = style1.replace("_", " ").title()
        style2_name = style2.replace("_", " ").title()
        
        new_prompts.append({
            "id": prompt_id,
            "name": f"{style1_name} {subject.capitalize()} to {style2_name}",
            "category": "artistic_style_transformation",
            "generation": {
                "text": f"A {subject} rendered in {style1.replace('_', ' ')} style with characteristic technique and visual treatment",
                "theme": style1,
                "model": "fast",
                "resolution": "1024x1024"
            },
            "edit": {
                "text": f"Transform to {style2.replace('_', ' ')} style with different artistic technique, rendering method, and visual aesthetic",
                "target_theme": style2,
                "transformation_type": f"{style1}_to_{style2}"
            }
        })
        prompt_id += 1
    
    print(f"Generated artistic style transformations: 70 prompts (IDs 381-{prompt_id-1})")
    
    # ========================================================================
    # COMPLEX MULTI-DIMENSIONAL TRANSFORMATIONS (451-500) - 50 prompts
    # ========================================================================
    
    complex_transforms = [
        ("Sunny Summer Park to Spooky Autumn Night", "location_time_mood_transformation",
         "sunny summer park with green grass, flowers, people relaxing, and cheerful atmosphere",
         "summer_lush",
         "spooky autumn night park with fallen leaves, fog, bare trees, eerie lighting, and mysterious atmosphere",
         "autumn_falling"),
        
        ("Modern Office to Medieval Castle Night", "era_location_time_transformation",
         "modern office with computers, fluorescent lights, contemporary furniture, and professional atmosphere",
         "contemporary_2000s",
         "medieval castle at night with torches, stone walls, moonlight, and historical atmosphere",
         "medieval_dark_ages"),
        
        ("Clear Day Beach to Stormy Night Beach", "weather_time_mood_transformation",
         "clear sunny day beach with bright light, calm waves, beachgoers, and relaxed atmosphere",
         "clear_sunny",
         "stormy night beach with dark clouds, crashing waves, lightning, and dramatic atmosphere",
         "thunderstorm"),
        
        ("Realistic City to Anime Fantasy City", "style_location_mood_transformation",
         "photorealistic modern city with detailed buildings, realistic lighting, and urban atmosphere",
         "photorealistic",
         "anime-style fantasy city with vibrant colors, magical elements, stylized architecture, and whimsical atmosphere",
         "anime_cel_shaded"),
        
        ("Victorian Garden to Cyberpunk Rooftop", "era_location_mood_transformation",
         "Victorian formal garden with ornate fountains, manicured hedges, classical statues, and elegant atmosphere",
         "victorian_1800s",
         "cyberpunk rooftop garden with neon plants, holographic displays, tech integration, and futuristic atmosphere",
         "cyberpunk_dystopia"),
    ]
    
    # Generate 50 complex multi-dimensional transformations
    for i in range(50):
        if i < len(complex_transforms):
            name, category, gen_text, gen_theme, edit_text, edit_theme = complex_transforms[i]
        else:
            # Generate more complex combinations
            base_locs = ["street", "building", "garden", "room", "plaza", "bridge", "market", "temple", "tower", "hall"]
            base_loc = base_locs[i % len(base_locs)]
            
            # Combine multiple dimensions
            time1 = ["day", "night", "dawn", "dusk"][i % 4]
            time2 = ["day", "night", "dawn", "dusk"][(i + 2) % 4]
            season1 = seasons[i % 4]
            season2 = seasons[(i + 2) % 4]
            mood1 = ["cheerful", "mysterious", "serene", "dramatic"][i % 4]
            mood2 = ["cheerful", "mysterious", "serene", "dramatic"][(i + 2) % 4]
            
            name = f"{season1.capitalize()} {time1.capitalize()} {base_loc.capitalize()} to {season2.capitalize()} {time2.capitalize()} {base_loc.capitalize()}"
            category = "location_time_seasonal_transformation"
            gen_text = f"{season1} {time1} {base_loc} with seasonal features, time-appropriate lighting, {mood1} mood, and atmospheric details"
            gen_theme = f"{season1}_scene"
            edit_text = f"{season2} {time2} {base_loc} with different seasonal characteristics, altered lighting, {mood2} mood, and transformed atmosphere"
            edit_theme = f"{season2}_scene"
        
        new_prompts.append({
            "id": prompt_id,
            "name": name,
            "category": category,
            "generation": {
                "text": gen_text,
                "theme": gen_theme,
                "model": "fast",
                "resolution": "1024x1024"
            },
            "edit": {
                "text": edit_text,
                "target_theme": edit_theme,
                "transformation_type": f"complex_multi_dimensional_{i+1}"
            }
        })
        prompt_id += 1
    
    print(f"Generated complex transformations: 50 prompts (IDs 451-{prompt_id-1})")
    print(f"\n✅ Total new prompts generated: {len(new_prompts)} (IDs 101-500)")
    
    return new_prompts


def main():
    """Main function to expand prompts"""
    
    print("="*80)
    print("EXPANDING PROMPT_THEME_100.JSON FROM 100 TO 500 PROMPTS")
    print("="*80)
    print()
    
    # Load existing prompts
    prompt_file = PROJECT_ROOT / "config" / "prompt_theme_100.json"
    
    with open(prompt_file, 'r') as f:
        data = json.load(f)
    
    existing_prompts = data['prompts']
    print(f"✓ Loaded {len(existing_prompts)} existing prompts")
    print()
    
    # Generate new prompts
    new_prompts = generate_new_prompts()
    print()
    
    # Combine prompts
    all_prompts = existing_prompts + new_prompts
    print(f"✓ Combined total: {len(all_prompts)} prompts")
    print()
    
    # Update metadata
    data['description'] = "500 diverse theme transformation prompts for comprehensive training"
    data['total_prompts'] = 500
    data['prompts'] = all_prompts
    
    # Update version
    data['version'] = "3.0"
    
    # Save back to file
    print("💾 Saving expanded prompts to file...")
    with open(prompt_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Successfully expanded {prompt_file.name} to 500 prompts!")
    print()
    print("Summary:")
    print(f"  - Original prompts: 1-100")
    print(f"  - Location transformations: 101-170 (70 prompts)")
    print(f"  - Era transformations: 171-240 (70 prompts)")
    print(f"  - Seasonal/Weather: 241-310 (70 prompts)")
    print(f"  - Mood transformations: 311-380 (70 prompts)")
    print(f"  - Artistic styles: 381-450 (70 prompts)")
    print(f"  - Complex multi-dimensional: 451-500 (50 prompts)")
    print(f"  - Total: 500 prompts")


if __name__ == "__main__":
    main()

