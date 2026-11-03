"""Reference vocabulary for server integration, not an eligibility filter.

Use building:use and building with the latest semantic-conflict rules. Preserve
mixed residential/nonresidential evidence; these sets do not implement matching,
facility masking or final owner qualification.
"""

RES={'apartments','bungalow','cabin','coach_house','detached','annexe','farm','ger','house','houseboat','residential','semidetached_house','static_caravan','stilt_house','terrace'}

NONRES={'commercial','industrial','office','retail','supermarket','warehouse','kiosk','religious','cathedral','chapel','church','mosque','synagogue','temple','school','university','college','kindergarten','hospital','clinic','civic','public','government','garage','garages','carport','shed','roof','barn','stable','greenhouse','outbuilding','service','parking','hangar','storage_tank','silo','transformer_tower','water_tower','hotel','motel','dormitory','barracks','prison','fire_station','police','police_station','train_station','transportation','bridge','stadium','sports_hall','sports_centre','grandstand','pavilion','toilets','cowshed','farm_auxiliary','car_repair','storage_rental','boathouse','power','military'}
