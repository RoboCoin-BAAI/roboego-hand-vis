"""Prepare a conservative draft for assistant review; does not edit annotations."""
import json
import re
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent
ROWS = json.loads((BASE / 'scene_inputs_01_39.json').read_text(encoding='utf-8-sig'))['groups']

# Category identifiers for the user's six top-level scene classes.
CATEGORY = dict(h='household', i='industrial_production', c='commercial_service',
                p='professional_work', u='public_space', o='other')


def draft(row):
    scene = (row['original_description'] or '').lower()
    task = (row['task_description'] or '').lower()
    text = scene + ' ' + task
    prefix = re.split(r'\b(?:for|where|with|used to)\b', scene, maxsplit=1)[0].strip(' .')
    def has(pattern):
        return bool(re.search(pattern, text))
    def result(code, area, reason):
        return dict(id=row['id'], category=CATEGORY[code], area=area, reason=reason)
    if has(r'\b(?:cat food|pet food|pet.s litter|feed\w* (?:the )?pet)\b') and has(r'household|domestic|\bhome\b'):
        return result('h', 'pet_care_area', 'Household pet feeding or litter-box care')
    if has(r'household|domestic') and has(r'\b(?:laundry|washing machine|mending|sewing|bathrobes?|clothing rack|hanging.*(?:towels?|clothes)|(?:clothes|towels?).*hanging)\b'):
        return result('h', 'laundry_area', 'Domestic clothes washing, drying or mending')
    if has(r'household|domestic') and has(r'\b(?:toiletries|mouthwash|personal care|hygiene products|soap dispenser|contact lens)\b'):
        return result('h', 'bathroom', 'Domestic personal-hygiene task')
    # Explicit places take priority over an activity-based inference.
    explicit = [
        (r'\bfactory\b|\bproduction line\b', 'i', 'factory_workshop'),
        (r'\bindustrial warehouse\b', 'i', 'industrial_warehouse'),
        (r'\bequipment room\b', 'i', 'equipment_room'),
        (r'\btesting lab\b', 'i', 'testing_lab'),
        (r'\bconstruction site\b', 'i', 'construction_site'),
        (r'\bconference center\b', 'p', 'conference_center'),
        (r'\bconference (?:room|table|area)\b|\bmeeting (?:room|area)\b', 'p', 'meeting_room'),
        (r'\bclassroom\b', 'p', 'classroom'),
        (r'\blaboratory\b|\bliquid samples\b', 'p', 'laboratory'),
        (r'\bhospital\b|\bclinic\b|\bhealthcare\b', 'p', 'healthcare_facility'),
        (r'\boffice (?:area|workspace|room)\b|\bin (?:the |an )?office\b', 'p', 'office'),
        (r'\bretail\b|\bsupermarket\b|\bshopping (?:cart|basket)\b|\brestocking\b|\bproduct (?:stocking|display|shelving)\b', 'c', 'retail_store'),
        (r'\brestaurant\b', 'c', 'restaurant'),
        (r'\bcafe\b|\bbeverage shop\b', 'c', 'beverage_shop'),
        (r'\bhotel\b', 'c', 'hotel'),
        (r'\bhair salon\b', 'c', 'hair_salon'),
        (r'\bbeauty salon\b', 'c', 'beauty_salon'),
        (r'\bpet shop\b|\bpet store\b', 'c', 'pet_shop'),
        (r'\bmassage room\b', 'c', 'massage_room'),
        (r'\bhome (?:gym|fitness|workout)\b', 'h', 'home_fitness_area'),
        (r'\bgym\b|\bfitness area\b|\bfitness equipment area\b', 'u', 'indoor_sports_venue'),
        (r'\bbadminton (?:court|net)\b', 'u', 'badminton_court'),
        (r'\bkitchen\b', 'h', 'kitchen'),
        (r'\bbathroom\b|\bwashstand\b', 'h', 'bathroom'),
        (r'\bbedroom\b|\bwardrobe\b', 'h', 'bedroom'),
        (r'\bliving room\b|\bliving area\b', 'h', 'living_room'),
        (r'\bdining (?:room|area|table)\b', 'h', 'dining_room'),
        (r'\bstudy (?:room|or desk)\b', 'h', 'study_room'),
        (r'\blaundry area\b', 'h', 'laundry_area'),
        (r'\bbalcony\b', 'h', 'balcony'),
    ]
    for pattern, code, area in explicit:
        if has(pattern):
            return result(code, area, 'Explicit scene or place-specific task: ' + pattern)
    if prefix in ('a packing area','a sorting area','a shelving area','a logistics area',
                  'a warehousing area','a warehouse area','a loading area','a cargo organization area'):
        return result('c', 'logistics_facility', 'Packing, sorting, shelving or cargo handling context')
    if has(r'\bshipping\b|\bexpress (?:delivery|sheet|box|boxes)\b|\bpackage label|\bpacking\b|\bpackaging cardboard\b'):
        return result('c', 'logistics_facility', 'Shipment/package handling task')
    if prefix in ('a household storage area','a domestic storage area','a storage room area'):
        return result('h', 'storage_room', 'Explicit household storage context')
    if prefix in ('an assembly area','an industrial assembly area','a workbench area') or has(
            r'\b(?:assemble|assembles|assembling|trims?|trimming|processing|fabricat\w*)\b.*\b(?:plastic (?:parts?|components?|pieces?)|copper wire|starter|motor|engine)\b'):
        return result('i', 'workroom', 'Small-component assembly or processing workbench; no factory-scale evidence')
    if has(r'\b(?:compressor|generator|starter motor|engine|motor repair)\b'):
        return result('i', 'workroom', 'Mechanical disassembly or repair workbench')
    if prefix == 'a quality inspection area':
        return result('i', 'workroom', 'Explicit product quality-inspection task')
    if has(r'\bcardboard (?:box|boxes|cartons?)\b') and has(r'\bpallet\b|\bscans?\b|\bscanning\b|\bseal\w*\b|\b(?:labels?|labeling|relabeling)\b|\bfold\w*.*\b(?:shape|tape|bottom)\b'):
        return result('c', 'logistics_facility', 'Carton preparation, shipment labeling, scanning or pallet handling')
    if has(r'\b(?:documents?|folders?|stapler|stapling|staples|paper punch|paper clips?|binder clips?|filing|binding|computer mouse|mouse pad|keyboard|office supplies|whiteboard|name tags)\b'):
        return result('p', 'office', 'Document, office-supply or computer-workstation task')
    if has(r'\bdesk\b|\bdesktop\b'):
        return result('p', 'office', 'Desk workstation context')
    if has(r'\b(?:fountain pen|dip pen|pen refill|ink cartridges?|ink bottle|writing instruments?|stationery|markers?|pencil|pencils|notebooks?|paper flyers|envelopes?|employee badges|work badges|ID (?:card|holder)|ID card|lanyards?.*check-in|sign-in board|writing on paper|writes? on paper)\b'):
        return result('p', 'office', 'Writing, stationery, correspondence or work-badge organization')
    if has(r'\b(?:organizing|collecting|sorts?|stacks?|arranges?|organizes|collects)\b.*\bpapers?\b') and not has(r'\b(?:scraps?|waste|shredded|torn|wrapping|crumpled|origami|colored paper|paper cups?|paper bags?)\b'):
        return result('p', 'office', 'Organization of paper documents')
    if has(r'\bworkspace\b') and has(r'\b(?:data cables?|earphone|electronic accessories|computer (?:case|tower|host)|electronic (?:equipment|devices)|computer peripherals|cables and adapters)\b'):
        return result('p', 'office', 'Electronic-equipment workstation organization or maintenance')
    if has(r'\b(?:shelf|rack) (?:column|post|upright)|\blabel.*\bshel(?:f|ves)\b'):
        return result('c', 'logistics_facility', 'Storage-shelf identification task')
    if has(r'\b(?:makeup|cosmetics?|eyeshadow|lipstick|powder compact|dressing table|jewelry|jewellery|necklace)\b'):
        return result('h', 'dressing_area', 'Personal makeup or accessory-care context, no salon/shop evidence')
    if has(r'\b(?:fitness equipment|exercise equipment|dumbbells?|barbells?|weight plates?|treadmill|elliptical|exercise bikes?|fitness (?:benches|bench|chairs?)|medicine ball|kettlebell|weightlifting equipment)\b'):
        return result('u', 'indoor_sports_venue', 'Exercise-equipment setup, storage or cleaning context')
    if has(r'household|domestic') and has(r'\b(?:toiletries|personal care|hygiene products|soap dispenser|soap bottles|mixing liquid cleaners|washing sinks?|wiping (?:a |the )?sink|scrubbing.*sinks?|washing a basin|contact lens|comb\b)'):
        return result('h', 'bathroom', 'Household personal-hygiene or washstand task')
    if has(r'\b(?:dishwasher|dishwashing|cutlery|tableware|drying rack|dishes|bowls?|plates?|pepper grinder|garlic|vegetables|cooking|cookware|kitchenware|kettle|refrigerator|fridge|bottle brush|baby bottles?|milk powder|coffee pot|tea set|coffee capsules|tea bags|utensils|glassware|drinking glasses|drinking vessels|ice maker|thermos cup)\b'):
        return result('h', 'kitchen', 'Food preparation or tableware cleaning/storage task')
    if has(r'\b(?:fold\w*|roll\w*)\b.*\b(?:towels?|garments?|clothing|clothes|laundry)\b|\b(?:towels?|garments?|clothing|clothes)\b.*\bfold\w*\b'):
        return result('h', 'laundry_area', 'Linen/clothing folding and organization')
    if has(r'\bshoe(?:s)?\b') and has(r'household|domestic|\bhome\b|shoe rack'):
        return result('h', 'shoe_care_area', 'Domestic shoe care/storage; doorway location is not established')
    if has(r'household|domestic') and has(r'\b(?:laundry|washing machine|mending|sewing|bathrobes?|clothing rack|hanging.*(?:towels?|clothes)|(?:clothes|towels?).*hanging)\b'):
        return result('h', 'laundry_area', 'Domestic clothes washing, drying or mending')
    if has(r'\b(?:fitness equipment|dumbbells?|barbells?|weight plates?|treadmill|elliptical|exercise bikes?|fitness benches|fitness bench|medicine ball|kettlebell)\b'):
        return result('u', 'indoor_sports_venue', 'Exercise-equipment setup, storage or cleaning context')
    if has(r'\b(?:cat food|pet food|feed\w* (?:the )?pet)\b') and has(r'household|domestic|\bhome\b'):
        return result('h', 'pet_care_area', 'Household pet-feeding context, no pet-shop evidence')
    if has(r'\b(?:books?|bookshelf|bookshelves|schoolbag|study materials)\b') and has(r'household|domestic|personal|residential'):
        return result('h', 'study_room', 'Domestic reading/study-material organization')
    if has(r'\b(?:game|chess|monopoly|jigsaw puzzle|puzzle pieces|toys)\b') and has(r'household|domestic|\bhome\b'):
        return result('h', 'home_recreation_area', 'Explicit household board-game context')
    return result('o', 'other', 'Descriptions do not distinguish a specific scene with sufficient confidence')


if __name__ == '__main__':
    decisions = [draft(row) for row in ROWS]
    (BASE / 'scene_decisions_draft.json').write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding='utf-8')
    print(Counter(d['category'] + '/' + d['area'] for d in decisions))
