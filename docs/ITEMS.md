# Items and prices

Fixed prices for all 75 catalog items, in `scenarios/public/items_example.json` (`market.schema.items`). The file also says which items an economy plays: the 24 with a **kind** below; nothing is drawn at random. With one file every economy plays these items; give a list of files (one per economy, as many as `repeats`) to play different items in each.

| kind | sold by | units for sale per market | items here |
|---|---|---|---:|
| monopoly | one seller (a different one per item) | 1.0 x buyers | 5 |
| oligopoly | min(3, half the sellers rounded up): 3 of 5 | 1.0 x buyers | 5 |
| scarce | every seller | 0.6 x buyers | 5 |
| competitive | every seller | 1.5 x buyers | 5 |
| balanced | every seller | 1.0 x buyers | 4 |

Every buyer values one unit of each item at the buyer value below (more units may be bought, but are worth nothing); every seller pays the seller cost per unit sold. Items marked *low grade* can also be delivered in a low grade (worth 0.4 x the value to the buyer, costing 0.5 x the cost). A buyer's cash is 1.15 x the sum, over all units for sale, of the midpoint of cost and value, divided by the buyers in the market. Items sold by some sellers sellers are dealt to the seller seats in file order; rotation gives every seller every seat's items once.

- seller cost: list price x U(0.5, 0.7), whole dollars, at least $1 (the same for every seller)
- buyer value: list price x U(1.2, 2.0), whole dollars, at least cost + $1 (the same for every buyer; for one unit of each item)
- sampled with seed 2026 from `scenarios/public/catalog.yaml`, then frozen

| id | item | list price | seller cost | buyer value | gain | kind | low grade |
|---|---|---:|---:|---:|---:|---|---|
| P01 | Satin Edge Rubber Grip Cuticle Pusher and Spoon Nail Cleaner, 8 Ounce | 0.79 | 1 | 2 | 1 |  |  |
| P02 | Q-tips Cotton Swabs, 500 Count | 1.35 | 1 | 3 | 2 |  |  |
| P03 | ForPro Toe Separators 12-ct. Assorted Colors | 1.67 | 1 | 2 | 1 | monopoly |  |
| P04 | World Pride Hot Lovely Metallic Sweet Lady Hollow Rose Flower Elast... | 1.72 | 1 | 3 | 2 |  |  |
| P05 | 10x Black Spiral Hair Pin Clip Bun Stick Pick for DIY Hair Style / ... | 1.80 | 1 | 3 | 2 |  |  |
| P06 | Swisspers Cotton Balls, Organic, 80 ct | 1.97 | 1 | 4 | 3 | competitive |  |
| P07 | SODIAL(R) 3000 Nail Art Gems Mixed Colours Shapes in Case (Size 2mm) | 2.23 | 1 | 4 | 3 |  |  |
| P08 | Hair Rollers - high-speed changing hair curlers Styling Rollers (16... | 2.59 | 1 | 4 | 3 |  |  |
| P09 | 500 Pcs Clear French Acrylic Style Artificial Half False Nails Nail... | 2.91 | 2 | 5 | 3 |  |  |
| P10 | 5pcs Nail Art Sable Acrylic Nail Brushes Pen Set with Cuticle Pushe... | 3.02 | 2 | 6 | 4 |  |  |
| P11 | Housweety Unique Dots Pattern Double Layer Cosmetic Bag Black | 3.08 | 2 | 5 | 3 |  |  |
| P12 | Body Fantasies Body Spray for Women, Cotton Candy Fantasy Fragrance... | 3.74 | 2 | 5 | 3 |  |  |
| P13 | Peppermint 100% Pure Therapeutic Grade Essential Oil- 10 ml | 3.75 | 3 | 7 | 4 |  |  |
| P14 | Tropic Isle Living Jamaican Black Castor Oil, 4 Ounce | 3.88 | 3 | 8 | 5 |  |  |
| P15 | NYX Matte Blotting Paper, White, 50 Count | 4.00 | 2 | 5 | 3 |  |  |
| P16 | Maybelline New York ColorSensational Lip Liner, Clear 10, 0.04 Ounce | 4.34 | 2 | 7 | 5 |  |  |
| P17 | Betty Dain Stylish Design Mold Resistant Shower Cap, The Fashionist... | 4.73 | 3 | 7 | 4 |  |  |
| P18 | Essie Nail Polish Marshmallow, .5 Ounce | 5.58 | 4 | 10 | 6 |  |  |
| P19 | Aquasentials Moisturizing Gloves | 5.69 | 4 | 7 | 3 | competitive |  |
| P20 | NYX Matte Lipstick, Nude | 5.88 | 4 | 9 | 5 |  |  |
| P21 | Adovia Natural Dead Sea Mud Soap - Great for Eczema, Psoriasis or A... | 6.00 | 4 | 9 | 5 | balanced |  |
| P22 | Essie Ridge Filler Base Coat, 0.46 oz | 6.12 | 4 | 10 | 6 | balanced | yes |
| P23 | Nivea Body Touch of Renewal Lathering Body Scrub, 12 Ounce | 6.46 | 3 | 10 | 7 |  |  |
| P24 | L'Oreal Paris Colour Caresse Wet Shine Lip Stain, Pink Resistance, ... | 6.67 | 4 | 11 | 7 | competitive |  |
| P25 | Lysol Touch of Foam Foaming Hand Soap, Creamy Vanilla Orchid, Pump,... | 6.68 | 4 | 13 | 9 |  |  |
| P26 | NYX Doll Eye Mascara, Extreme Black, Volume, DE02, 0.28 oz | 7.14 | 4 | 11 | 7 |  |  |
| P27 | Maybelline New York Eye Studio Lasting Drama Gel Eyeliner, Eggplant... | 7.57 | 5 | 12 | 7 | balanced | yes |
| P28 | Sunburnt Therapeutic After Sun Relief, 6-Ounce Tube | 7.85 | 5 | 10 | 5 | scarce |  |
| P29 | NYX Cosmetics Eye Shadow Palette 10 Color, Jazz Night, 0.49-Ounce | 7.97 | 4 | 10 | 6 |  |  |
| P30 | elf Studio High Definition Loose Face Powder | 8.98 | 5 | 16 | 11 | oligopoly |  |
| P31 | Vivo Per Lei Moisturizing Day Cream, 1.7-Fluid Ounce | 8.99 | 6 | 12 | 6 | monopoly | yes |
| P32 | Banana Boat Sport Cool Zone Sunscreen Continuous Spray SPF 50, 6 Fl... | 9.54 | 6 | 15 | 9 |  |  |
| P33 | Alpha Hydrox Enhanced Lotion 10 percent Glycolic AHA -- 6 fl oz Ant... | 9.67 | 6 | 13 | 7 |  |  |
| P34 | CND Cuticle Remover - 6oz | 9.70 | 7 | 17 | 10 |  |  |
| P35 | Pink Armor Nail Growth Formula Treatments, 0.50 Fluid Ounce | 9.80 | 6 | 15 | 9 |  |  |
| P36 | Cinema Secrets Stainless Steel Makeup Palette, Small | 9.95 | 7 | 20 | 13 |  |  |
| P37 | L. Erickson Ponytail Holders - Set of Eight - Purple | 10.00 | 6 | 16 | 10 |  |  |
| P38 | SHANY Cosmetics All Black Light Weighted Makeup Train Case, Moonles... | 10.01 | 6 | 14 | 8 |  |  |
| P39 | TIGI Bed Head Hard Head Hair Spray, 10.6 Ounce | 10.02 | 7 | 16 | 9 |  |  |
| P40 | Sex Bomb Bath Bomb by LUSH | 10.76 | 6 | 21 | 15 | competitive |  |
| P41 | Qtica Intense Cuticle Repair Balm | 12.31 | 7 | 18 | 11 |  |  |
| P42 | Sleep Mask with Ear Plugs - This Eye Mask Is for Sleeping Better An... | 12.95 | 7 | 16 | 9 |  |  |
| P43 | Color Oops Hair Color Remover Extra Strength 1 Application | 13.10 | 8 | 22 | 14 |  |  |
| P44 | Mary Kay Foundation Primer Sunscreen Broad Spectrum SPF 15 1 fluid ... | 13.25 | 8 | 21 | 13 |  |  |
| P45 | Instant Heat Curling Iron, 1-Inch | 13.95 | 8 | 27 | 19 |  |  |
| P46 | Conair 1875 Watt Ionic Conditioning Hair Dryer | 14.49 | 8 | 23 | 15 |  |  |
| P47 | Zoya Remove Plus Nail Polish Remover in Big Flipper 8 oz | 15.74 | 9 | 29 | 20 | scarce |  |
| P48 | Luxury Spa Bath Pillow Large | 16.95 | 11 | 28 | 17 | monopoly |  |
| P49 | SHANY Cosmetics Feather Hair Extension Kit, 8 Ounce (Feather Hair, ... | 16.99 | 12 | 32 | 20 |  |  |
| P50 | California Baby Bubble Bath Aromatherapy, 13 oz (Eucalyptus ease (f... | 17.50 | 10 | 34 | 24 |  |  |
| P51 | Gold N' Hot GH3013 Gold Tone Crimping Iron, 2 Inch | 18.99 | 10 | 33 | 23 |  |  |
| P52 | Proactiv Solution JUMBO Super Size Clarifying Night Cream 2.5oz | 19.50 | 13 | 31 | 18 |  |  |
| P53 | (1 oz / 30 ml) GLYCOLIC Acid 70% Skin Chemical Peel - Unbuffered - ... | 19.98 | 12 | 39 | 27 |  |  |
| P54 | Olay Regenerist Microdermabrasion &amp; Peel System 1 Kit | 21.99 | 15 | 41 | 26 |  |  |
| P55 | Nu Derm Healthy Skin Protection SPF 35 - 90ml/3oz | 22.50 | 13 | 28 | 15 | oligopoly |  |
| P56 | Travel Packing Cubes- 4pc Set With Free &hearts; Shoe bags&hearts; ... | 22.95 | 13 | 37 | 24 | oligopoly |  |
| P57 | Trind Natural Nail Repair | 23.28 | 12 | 42 | 30 |  |  |
| P58 | Jerdon J1010 6-Inch Portable Tabletop Two-Sided Swivel Lighted Make... | 23.79 | 14 | 37 | 23 | monopoly | yes |
| P59 | Equate - Hair Regrowth Treatment for Women with Minoxidil 2%, 3 Mon... | 24.00 | 12 | 29 | 17 |  |  |
| P60 | Redken All Soft Shampoo and Conditioner Set 10.1 Oz 8.5 Oz | 26.66 | 16 | 52 | 36 | oligopoly |  |
| P61 | SHANY UV Gel Light Nail Dryer, 36 Watts Pro Series | 26.69 | 14 | 49 | 35 |  |  |
| P62 | NYX Cosmetics Waiting for Tonight 46 Colors Makeup Kit Palette with... | 26.97 | 15 | 46 | 31 | oligopoly | yes |
| P63 | Paul Mitchell Tea Tree Hair And Scalp Treatment Unisex, 16.9 Ounce | 27.00 | 18 | 53 | 35 | scarce |  |
| P64 | BareMinerals Original Foundation Broad Spectrum SPF 15 8 g/0.28 Oz ... | 28.00 | 18 | 48 | 30 |  |  |
| P65 | Cure Natural Aqua Gel 250ml - Best selling exfoliator in Japan! | 28.00 | 15 | 41 | 26 | balanced | yes |
| P66 | Fogless Shower Mirror with Squeegee by ToiletTree Products. Guarant... | 29.95 | 18 | 54 | 36 |  |  |
| P67 | Tree Hut Shea Body Butter, Brazilian Nut, 7-Ounce (Pack of 3) | 30.95 | 21 | 49 | 28 | competitive |  |
| P68 | WEN by Chaz Dean Lavender Cleansing Conditioner 16 oz | 31.50 | 20 | 41 | 21 | scarce |  |
| P69 | Viva La Juicy By Juicy Couture For Women Eau De Parfum Spray 1.7 Oz | 33.00 | 18 | 55 | 37 |  |  |
| P70 | Lactic Acid 50% Gel Peel - Enhanced with Kojic Acid &amp; Bearberry... | 34.00 | 24 | 65 | 41 |  |  |
| P71 | Tinted Moisturizer - Almond | 36.00 | 22 | 46 | 24 | scarce |  |
| P72 | TCA 15% Gel Peel - Salicylic Acid 5% Enhanced with Botanical Extrac... | 39.95 | 25 | 78 | 53 | monopoly |  |
| P73 | SHANY COSMETICS The Masterpiece 7 Layers All-in-One Makeup Set | 41.89 | 28 | 81 | 53 |  |  |
| P74 | Revlon RVSP3501B1 Moisturestay Quick Heat Paraffin Bath | 45.99 | 26 | 70 | 44 |  |  |
| P75 | Dr. Dennis Gross Skincare Extra Strength Alpha Beta Peel, 30 Count JAR | 74.40 | 49 | 120 | 71 |  |  |
