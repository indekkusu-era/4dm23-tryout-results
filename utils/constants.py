beatmaps = """exnoiz - non amen girl [loading...]
NanoriitaP - Ideal Picture [Overshadow 1.2x (162bpm)]
sky_delta - Lazy Addiction [Stage 5: Tidur]
Megurimeguru - Completion Feat. Hatsune Miku [Beatmap's Complete World (244bpm)]
Streetlight Manifesto - Ungrateful [mbop mbop]
Adust Rain - psychology [despondent 1.05x (161bpm)]
Xi - Event Hor!zon [FelixSpade's LN Prodigy (cut)]
Memme feat. M2U - Sky of the Ocean [Logan x HowToPlayLN's Troposphere x1.1]
Gardens - Rafflesia [Malpighiales (edit)]
Teikyou - Deadly Slot Game [Jackpot!]
Think Chan - Four Seasons [SV Seasons]"""

weights = {
    'Consistency': 15,
    'LN': 6,
    'Hybrid': 9,
    'Consistency2': 6,
    'Tech': 6,
    'Wildcard': 12,
    'LN Density': 6,
    'LN Release': 6,
    'Hybrid.1': 9,
    'SV Early': 6,
    'SV Late': 6
}.values()

weights = dict(zip(beatmaps.split("\n"), weights))
