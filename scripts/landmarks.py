# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "shapely"]
# ///
"""Named places of Ede around the Enka-terrein: style overrides and extra geometry for buildings, plus the viewer's
place list.

Sources: OpenStreetMap (names and positions), AHN (heights), BOEi (boei.nl, the foundation that restored the Enka
monuments). Places without a `text` show only their name and kind.

Output: data/build/landmarks.json
"""

import json

import numpy as np

from config import BUILD, RAW, in_era
from geo import wgs_to_rd

# Buildings found by an address or point; `pand` is resolved to the 3D BAG building under that point.
BUILDINGS = [
    {
        "name": "Station Ede Centrum",
        "at": (174277, 450585),
        "kind": "Station · 1902",
        "style": "station1902",
        "wall": (150, 78, 56),
        "roof": (62, 66, 74),
        "frame": (238, 232, 218),
        "text": "Het kleine station midden in Ede, geopend op 1 mei 1902 aan de lijn naar Barneveld en Nijkerk, die de eiermarkt en de pluimveebedrijven van Ede verbond met de haven van Nijkerk en de lijn Utrecht-Arnhem. Het stationsgebouw uit 1902 is een rijksmonument; het werd in 1973 buiten gebruik gesteld en was daarna lang het Historisch Museum Ede. Hier stopt de Valleilijn, op weg tussen Ede-Wageningen en Barneveld.",
        "more": "https://nl.wikipedia.org/wiki/Station_Ede_Centrum",
    },
    # The Enka-terrein: the rayon factory of the Eerste Nederlandsche Kunstzijdefabriek, now a new neighbourhood.
    {
        "name": "Spinnerijschoorsteen",
        "at": (175084, 448248),
        "kind": "Schoorsteen · 1927",
        "text": "De dikke, zoals de arbeiders hem noemden: de schoorsteen van de spinnerij, de hoogste gemetselde schoorsteen van Nederland. Hij voerde de spingassen af, door buizen over de daken en langs de gevels van de spinnerij en de bleekgebouwen; zolang er geen afzuiging was, kregen de spinners er pijnlijke ogen van. In mei 1940 zouden de schoorstenen worden gesloopt, omdat ze voor het Duitse leger een te duidelijk baken waren, maar de snelle inval maakte daar een eind aan. BOEi restaureerde hem in 2020; nu ligt hij midden op het Schoorsteenplein, en er broedt een torenvalk in.",
        "more": "https://www.boei.nl/projecten/schoorstenen-enka-ede",
    },
    # The old canteen, a rijksmonument: a school below, after-school care and a nursery on the top floor.
    {
        "name": "Kantinegebouw",
        "at": (175018, 448454),
        "kind": "Rijksmonument · brede school",
        "style": "canteen",
        "wall": (196, 194, 186),
        "roof": (70, 72, 76),
        "frame": (58, 128, 104),
        # The upper floor runs on past the ground floor at both ends, floating on columns over the open space below
        # (AM, Op Enka, p. 33, and photos): in 3D BAG those are the roof parts of about 11 m.
        "stilts": {"lift": (10.5, 11.5), "under": 6.8},
        "text": "J.C. Hartogs, die de fabriek in 1919 bouwde, liet het kantinegebouw na de Tweede Wereldoorlog neerzetten, als een vrolijke plek die zijn medewerkers de oorlog moest doen vergeten: licht, ruim en met art-decodetails, de bovenverdiepingen zwevend op kolommen. BOEi restaureerde het eerst als casco en maakte er daarna een bijna energieneutrale brede school van: in de onderste twee lagen basisschool De Ontdekking, met het schoolplein aan de westkant, en bovenin buitenschoolse opvang en kinderopvang, met een terras. Het project won de Erfgoed Duurzaamheidsprijs 2024.",
        "more": "https://www.boei.nl/projecten/schoorstenen-enka-ede",
    },
    {
        "name": "Bitterzoutloods",
        "at": (174877.1, 448385.3),
        "kind": "Rijksmonument · 1919",
        "style": "loods",
        "wall": (124, 66, 50),
        "roof": (98, 100, 104),
        "frame": (232, 232, 226),
        "text": "De loods waar de fabriek haar bitterzout (magnesiumsulfaat) bewaarde, een van de rijksmonumenten van het Enka-terrein. In november 2015 begon Wonen op Enka, van AM en a.s.r., aan de verbouwing tot vijftien unieke loftwoningen in het historische gebouw; in december 2016 kwamen de eerste bewoners, en werd een monument van de oude kraanbaan onthuld.",
        "more": "https://www.am.nl/wp-content/uploads/2016/12/Persbericht-Loftwoningen-in-Bitterzoutloods-Op-Enka-in-Ede-bewoond.pdf",
    },
    {
        "name": "Kantine van de kazerne",
        "at": (174956.3, 449561.6),
        "kind": "Keuken- en kantinegebouw · 1963",
        "text": "Het voormalige keuken- en kantinegebouw van de kazernes, uit 1963. Hier zat later ook een tijd de Heidebrouwerij.",
    },
    {"name": "Gymlokaal Enka", "since": 2021, "at": (174977, 448483), "kind": "Gymzaal"},
    {
        "name": "Westhal",
        "at": (174912, 448571),
        "kind": "Rijksmonument · 1922",
        # One building in 3D BAG with the Poortgebouw: dark brick, red window frames.
        "style": "factory",
        "wall": (118, 66, 52),
        "roof": (92, 94, 98),
        "frame": (170, 34, 32),
        "text": "Een fabriekshal van zo'n 10.000 m², sinds 2014 van BOEi. Lang was het een evenementenlocatie, met autoshows, rommelmarkten en opnames voor televisie, tot dat om veiligheidsredenen stopte. Daarna vestigde Accell er De Fietser, een fietsbelevingscentrum. Die wilde ook op zondag open, en de gemeente paste daarvoor de winkelverordening aan.",
        "more": "https://www.am.nl/wp-content/uploads/2024/03/OP-Enka-waar-elke-steen-een-verhaal-vertelt.pdf",
    },
    # The main building of the factory, with the arch and the letters ENKA: in 3D BAG part of the Westhal's building.
    {
        "name": "Poortgebouw",
        "at": (174985, 448604),
        "kind": "Rijksmonument · 1922",
        "text": "Het hoofdgebouw van de fabriek, met de grote boog waar je het terrein op ging en bovenin de letters ENKA. Toen de bouw van de wijk begon, had de Grondbank er haar kantoor, en regelde de hoofduitvoerder van daaruit ook de evenementen in de leegstaande Westhal. Bouwbedrijf Kreeft restaureerde het voor BOEi.",
        "more": "https://www.am.nl/wp-content/uploads/2024/03/OP-Enka-waar-elke-steen-een-verhaal-vertelt.pdf",
    },
    # The new Ketelhuis next to the boiler chimney, after the boiler house that stood there.
    {
        "name": "Het Ketelhuis (Stoer)", "since": 2020,
        "at": (175089.9, 448445.8),
        # Near-black brick with yellow panels beside the windows and grey shutters.
        "style": "stoer",
        "wall": (46, 46, 50),
        "roof": (70, 72, 76),
        "frame": (232, 186, 40),
        "kind": "Woongebouw · 2020",
        "text": "Een nieuw gebouw, geïnspireerd op het ketelhuis dat hier naast de dunne schoorsteen stond, met een groot grasveld eromheen en een speeltuin met schommels en een hindernisbaan. Op de begane grond woont Stoer: ouders van jongvolwassenen met een verstandelijke beperking richtten de stichting op, zodat hun kinderen ieder een eigen appartement hebben, met zorg en gezamenlijke ruimtes. Ze mochten zelf een kruisje op de kaart van de wijk zetten. Boven wonen huurders van Woonstede.",
        "more": "https://www.am.nl/wp-content/uploads/2024/03/OP-Enka-waar-elke-steen-een-verhaal-vertelt.pdf",
    },
    {
        "name": "De Reehorst", "at": (174497, 448446), "kind": "Congrescentrum en hotel",
        "text": "In 1932 geopend als ontspanningsgebouw voor het personeel van de Enka-fabriek aan de overkant van de weg, met een theaterzaal waar ook films werden gedraaid. Nu is het een congrescentrum en theater, met een hotel, en een plek voor bedrijfsfeesten. De naam leeft voort in de Reehorsterweg en in Edese sportverenigingen die teruggaan op de omnivereniging Sportclub Reehorst.",
        "more": "https://nl.wikipedia.org/wiki/De_Reehorst_(Ede)",
    },
    # The kazerneterrein, from OpenStreetMap (all tagged as municipal heritage).
    {
        "name": "Johan Willem Friso kazerne",
        "at": (174746, 448948),
        "kind": "Rijksmonument · 1904",
        "text": "Ede werd rond 1900 garnizoensplaats, om de oefenterreinen en de bouwgrond. Als eerste kwamen in 1904-1906 twee infanteriekazernes naast elkaar, Infanteriekazerne I en II; in 1934 kregen ze de namen van de prinsen Maurits en Johan Willem Friso. Hier lag het 11e, later 22e Regiment Infanterie. Het neorenaissance-hoofdgebouw, waarschijnlijk van ingenieur Van Stolk, is een rijksmonument. Sinds 2014 zit er Akoesticum, een oefencentrum voor de podiumkunsten.",
        "more": "https://nl.wikipedia.org/wiki/Johan_Willem_Frisokazerne_(Ede)",
    },
    {
        "name": "Mauritskazerne",
        "at": (174712, 449149),
        "kind": "Rijksmonument · 1904",
        "text": "De tweeling van de Johan Willem Frisokazerne, tot 1934 Infanteriekazerne I. Bij de mobilisaties voor beide wereldoorlogen kwamen er uit het hele land soldaten naar Ede; achter de kazerne verrees een enorm tentenkamp. In 1994 gaf het complex van vijf kazernes hier zijn naam: Prins Mauritskazerne. Na het vertrek van Defensie eind 2010 werd het hoofdgebouw een plek voor vergaderingen en evenementen.",
        "more": "https://nl.wikipedia.org/wiki/Mauritskazerne",
    },
    {
        "name": "Arthur Kool kazerne",
        "at": (174763, 449603),
        "kind": "Rijksmonument · 1906-1908",
        "text": "Een artilleriekazerne in chaletstijl, H-vormig en bijna het spiegelbeeld van de Van Essenkazerne ernaast. Eerst lag hier het 1e Regiment Huzaren, na de Eerste Wereldoorlog de 2e Artillerie Brigade. In de oorlog gebruikte de Waffen-SS hem als opleidingskazerne en heette hij met zijn buren Bismarck-Kaserne. Tot 1968 werden hier artilleristen opgeleid.",
        "more": "https://nl.wikipedia.org/wiki/Arthur_Koolkazerne",
    },
    {
        "name": "P.L. Bergansiuskazerne",
        "at": (174821, 449534),
        "kind": "Rijksmonument · 1936",
        "text": "Een U-vormige kazerne in zakelijk-expressionistische stijl met trekjes van de Amsterdamse School, in 1936 geopend voor de 120 leerlingen van de School Reserve Officieren Bereden Artillerie. In de oorlog zat hier de Ortskommandantur van de bezetter, later een opleiding van de Waffen-SS; na de bevrijding waren hier tot november 1945 Canadezen gelegerd.",
        "more": "https://nl.wikipedia.org/wiki/P.L._Bergansiuskazerne",
    },
    {
        "name": "Van Essenkazerne",
        "at": (174952, 449617),
        "kind": "Rijksmonument · 1906-1908",
        "text": "Het spiegelbeeld van de Arthur Koolkazerne, in chaletstijl, voor het 4e Regiment Veld Artillerie. Bij de mobilisatie van 1914 groeide het aantal soldaten in Ede tot zo'n 8000, met een tentenkamp en inkwartiering bij de Edenaren; in 1939 kwamen er ook meer dan duizend paarden.",
        "more": "https://nl.wikipedia.org/wiki/Van_Essenkazerne",
    },
    {
        "name": "Elias Beeckman kazerne",
        "at": (175356, 449896),
        "kind": "Rijksmonument · 1939",
        "text": "Een infanteriekazerne uit het bouwprogramma van 1938, van de genie-kapitein A.G. Boost. Het 22e Regiment Infanterie vertrok hier in 1940 naar de Grebbeberg. Na de oorlog werden hier bataljons voor Nederlands-Indië opgeleid, later de verbindingstroepen. In 1966 werd hier de VVDM opgericht, de eerste vakbond voor dienstplichtigen ter wereld. In 2010 vertrok Defensie; de wijk Veluwse Poort kwam ervoor in de plaats, met de monumenten erin.",
        "more": "https://nl.wikipedia.org/wiki/Elias_Beeckmankazerne",
    },
    # Churches and the old centre.
    {
        "name": "Oude Kerk", "at": (174507, 450892), "kind": "Kerk · rijksmonument",
        "text": "De kerk van Johannes de Doper, rond 1200 begonnen als kleine stenen hallenkerk, waarschijnlijk op de plek van een houten kerkje. In 1421 werd ze door oorlogshandelingen verwoest en daarna herbouwd; het koor is uit die tijd. In 1635 sloeg de bliksem in de toren, die helemaal uitbrandde; de spits viel op het dak en het herstel duurde jaren. In de toren zijn nog consoles te zien van de beelden van Johannes de Doper en Sint-Barbara, die er voor de beeldenstorm stonden. Een plaquette op de toren herinnert sinds 1946 aan de Tweede Wereldoorlog.",
        "more": "https://nl.wikipedia.org/wiki/Oude_Kerk_(Ede)",
    },
    {
        "name": "Sint Antonius van Padua", "at": (174422, 449193), "kind": "Kerk · 1953",
        "text": "De parochie is begonnen met de katholieke militairen van het garnizoen van Ede. De eerste kerk, uit 1920, werd in 1944 verwoest bij geallieerde bombardementen voor Market Garden. De huidige kerk, een basiliek met een breed torenfront en een groot roosvenster in glas-in-lood, is gebouwd in 1952-1953 naar ontwerp van J.M. van Hardeveld en op 21 juli 1954 gewijd. Het orgel is van Pels (1966), de beelden van Maria en Jozef (1961) zijn van Wim Harzing.",
        "more": "https://nl.wikipedia.org/wiki/Sint-Antonius_van_Paduakerk_(Ede)",
    },
    {
        "name": "Historisch Museum Ede", "at": (174213, 450948), "kind": "Museum",
        "text": "Vereniging Oud Ede, in 1924 opgericht, verzamelde de collectie: gereedschap, klederdracht, foto's en documenten over Ede. Het museum zat sinds 1937 in een oud boerderijtje aan de Driehoek en later in het stationsgebouw van Ede Centrum. Toen de eigenaar daar de huur sterk verhoogde, verhuisde het in januari 2016 naar de tweede verdieping van Cultura, waar het wisselende exposities houdt en nog zoekt naar een eigen pand.",
        "more": "https://nl.wikipedia.org/wiki/Historisch_Museum_Ede",
    },
    {
        "name": "Kasteel Hoekelum", "at": (174878, 447502), "kind": "Kasteel · rijksmonument",
        "text": "In 1396 werd Randolph van Hoekelum, jagermeester van de hertog van Gelre, beleend met 'het oude goed te Hoekelum'; het oudste stenen gebouw is van omstreeks 1325. Het goed ging door veel handen, onder meer de families Van Poelwijck, Van Lawick (die het in 1695 kocht) en Van Wassenaer. Rond 1735 werd het kasteel bijna helemaal herbouwd, rond 1914 verbouwd naar ontwerp van J.W. Hanrath. Het landgoed van 112 hectare, met een landschapspark in de stijl van Zocher, is sinds 1988 van Het Geldersch Landschap; nu is het een trouwlocatie van de gemeente.",
        "more": "https://nl.wikipedia.org/wiki/Kasteel_Hoekelum",
    },
    {
        "name": "De Keetmolen", "at": (174438.0, 449084.4), "kind": "Molen · rijksmonument",
        "text": "Een achtkante stenen beltmolen aan de Stationsweg, door de wind gedraaid op de bovenkruier. Het opschrift noemt 1750, maar volgens het kadaster is hij in 1857 gebouwd. De naam komt van de werkkeet van de grondwerkers die tussen 1845 en 1878 aan de spoorlijn Arnhem-Utrecht werkten. Hij maalde rogge, boekweit en later gerst, voor de familie Van der Craats, die hem van 1860 tot 1971 bezat. In de jaren vijftig hield het malen op; sinds 18 augustus 1971 is de molen van de gemeente, die hem in 1978 liet restaureren. Vrijwilligers laten hem nog draaien. De wieken hebben een vlucht van 24 meter.",
        "more": "https://nl.wikipedia.org/wiki/De_Keetmolen_(Ede)",
    },
    {
        "name": "Ulu Moskee", "since": 2016, "at": (173438.7, 449109.6), "kind": "Moskee · 2016",
        "style": "mosque",
        "wall": (228, 224, 214),
        "roof": (90, 110, 120),
        "text": "De Turkse moskee van Ede (Ulu Camii). De gemeenschap zat eerst in een bedrijfspand aan de Frankeneng en zocht met de gemeente een plek voor een nieuwe moskee. In december 2012 koos de gemeenteraad de Verlengde Blokkenweg, waar een nieuw bestemmingsplan voor nodig was. De nieuwe moskee, zo'n 2500 m², was begin december 2016 klaar.",
        "more": "https://mosqpedia.org/en/mosque/483",
    },
    {
        "name": "Pathé Ede", "since": 2000, "at": (172913, 447516), "kind": "Bioscoop · 2000",
        "style": "cinema",
        "wall": (178, 34, 38),
        "text": "Voorheen Cinemec: een bioscoop in de geluidswal langs de A12, bij de afslag Ede. Acht rode metalen dozen, de zalen, steken uit het gras op; de hal is uitgesneden uit de wal, en bij de tweede bouwfase kwamen een evenementenzaal en een parkeergarage in de wal. Doordat de zalen zwaar zijn geïsoleerd, vormen ze zelf een stuk van het geluidsscherm langs de snelweg. Het was de eerste bioscoop van Nederland met het A-label voor duurzaamheid en won Edese architectuurprijzen in 2006 en 2009. Pathé nam de bioscoop in 2014 over van Cinemec.",
        "more": "https://www.dp6.nl/en/projects/cinemec",
    },
    {
        "name": "Stadspoort", "since": 1995, "at": (173046, 447746), "kind": "Winkelcentrum · 1995",
        "text": "Een gedeeltelijk overdekt wijkwinkelcentrum uit 1995, als een poort over de weg de stad in, tussen de wijken Rietkampen en Maandereng. Met een bibliotheek, een Jumbo en een Albert Heijn is het ook een sociaal en cultureel middelpunt voor de buurt. In 2017 liet eigenaar Multi er discounter Action in komen om leegstand op te lossen.",
        "more": "https://multi.eu/news/multi-netherlands-welcomes-action-to-stadspoort-shopping-centre-in-ede-and-resolves-vacancy-issue/",
    },
    {
        "name": "Koningin Beatrixschool", "since": 1985, "at": (173322, 447835), "kind": "Basisschool · CNS",
        "text": "Een christelijke basisschool van CNS Ede in de Maandereng, met in 2025 ruim 130 leerlingen. Een gang verbindt het gebouw met de twee andere scholen van dit schoolcomplex, Uniek en De Franciscus.",
        "more": "https://www.beatrix.cnsede.nl/2026-2027-schoolgids/",
    },
    {
        "name": "Uniek", "since": 1985, "at": (173305, 447786), "kind": "Basisschool · openbaar",
        "text": "Openbare basisschool van stichting Proominent, met zo'n negentig leerlingen. In het gebouw zit ook de Peuterpoort, een peuterspeelzaal van Spelenderwijs Ede.",
        "more": "https://allecijfers.nl/basisschool/uniek-ede",
    },
    {
        "name": "De Franciscus", "since": 1985, "at": (173262, 447783), "kind": "Basisschool",
        "text": "De derde school van het schoolcomplex in de Maandereng, aan de Mesdagstraat; een overdekte gang verbindt haar met de Koningin Beatrixschool en Uniek. Naast de scholen ligt de sportzaal De Maandereng.",
        "more": "https://www.beatrix.cnsede.nl/2026-2027-schoolgids/",
    },
    {
        "name": "Ziekenhuis Gelderse Vallei", "since": 2000, "at": (172764.9, 448303.8), "kind": "Ziekenhuis · 2000",
        "text": "In 1987 fuseerden het Julianaziekenhuis in Ede, het streekziekenhuis in Bennekom en het Pieter Pauwziekenhuis in Wageningen; in 1990 voegde het ziekenhuis van Veenendaal (in Rhenen) zich erbij. In 1992 werd een nieuw ziekenhuis in Rietkampen goedgekeurd, dat in 2000 opende als Ziekenhuis Gelderse Vallei.",
        "more": "https://nl.wikipedia.org/wiki/Ziekenhuis_Gelderse_Vallei",
    },
    {
        "name": "Riedel", "at": (172855.4, 449291.0), "kind": "Drankenfabriek",
        "text": "De grootste producent van vruchtensappen en vruchtendranken van Nederland, met merken als Appelsientje, CoolBest, DubbelFrisss, Taksi en Extran. Het bedrijf zegt sinds 1879 mineraalwater, limonade en vruchtendranken te maken. Van 1970 tot 2017 was het van FrieslandCampina; op 10 november 2017 nam Standard Investment het over. In 1973 bracht Riedel de eerste vruchtensappen in pakken uit, onder de naam Appelsientje. In 2016 waren er zo'n 200 medewerkers en een omzet van ongeveer 125 miljoen euro.",
        "more": "https://www.foodingredientsfirst.com/news/standard-investment-completes-frieslandcampina-riedel-takeover.html",
    },
    {
        "name": "AH XL", "at": (172862.7, 449167.6), "kind": "Supermarkt",
        "text": "Hier stond de Berko, een groot overdekt winkelcentrum dat op 30 januari 1978 afbrandde. Later kwam hier een Maxis, een hypermarkt van de Bijenkorf-groep, die tot 2002 bleef; daarna kwam er een extra grote C1000. Op 16 juli 2008 werd het een AH XL, met zo'n 3100 m² een van de grootste van Albert Heijn.",
        "more": "https://nl.wikipedia.org/wiki/Maxis_(hypermarkt)",
    },
    # The Stinger dome: a white ball of a simulator for anti-aircraft missiles, which 3D BAG draws as a block.
    {
        "name": "Stingerbol",
        "at": (174928, 448887),
        "kind": "Simulator · 1984",
        "style": "dome",
        "wall": (236, 236, 232),
        "roof": (236, 236, 232),
        "sphere": {"r": 9.6, "top": 18.0},
        "text": "Een witte bol uit 1984 waarin militairen oefenden met het afvuren van Stinger-luchtdoelraketten: een simulator die doelen op de binnenkant van de koepel liet vliegen. Nu exploiteert de Stichting Stingerbol hem als plek voor evenementen, en is hij een van de markante herinneringen aan het militaire verleden van dit terrein, waar het World Food Center komt.",
    },
    {"name": "50PK", "at": (174984, 449488), "kind": "Galerie"},
]

# Points without a building of their own.
POINTS = [
    {
        "name": "P+R-garage",
        "at": (174658.7, 448644.7),
        "since": 2026,
        "kind": "Parkeergarage · 2026",
        "text": "De nieuwe betaalde P+R-garage bij station Ede-Wageningen, gebouwd als deel van de vernieuwing van het station; volgens de gemeente open sinds 1 augustus 2026. De hoogte op de kaart is geschat: het gebouw is nieuwer dan de hoogtemetingen.",
    },
    # The north-east corner of the Carré, the main factory block, that was left standing.
    {
        "name": "Hoek van het Carré",
        "at": (175210, 448526),
        "kind": "Fabrieksmuur · 1922",
        "text": "Een hoek van het Carré, het grote blok van de fabriek, die bleef staan toen de rest werd gesloopt. Van de noordwand bleef ook een groot deel over, als losse muur met lege ramen; tegen de binnenkant zijn later de muurwoningen gebouwd.",
    },
    # The largest barracks of Ede, demolished for the new neighbourhood but for a few buildings.
    {
        "name": "Simon Stevinkazerne",
        "at": (175998, 450212),
        "kind": "Kazerne · 1939",
        "text": "De grootste kazerne van Ede, uit het bouwprogramma van 1938, voor het 12e Regiment Motor Artillerie, dat in 1940 vocht in de Slag om de Grebbeberg. De bezetter liet hem afbouwen, met uitkijktorens en schuilkelders, en noemde hem Langenbergkaserne. Later lag hier het Regiment Verbindingstroepen. Na het vertrek van Defensie in 2010 ging het meeste tegen de grond voor de nieuwe wijk; in de jaren rond 2009 waren hier nog de gebouwen uit de jaren 70 en 80.",
        "more": "https://nl.wikipedia.org/wiki/Simon_Stevinkazerne",
    },
    # Not in 3D BAG; drawn from CHIMNEYS below.
    {
        "name": "Ketelschoorsteen",
        "at": (175119.4, 448434.5),
        "kind": "Schoorsteen · 1920",
        "text": "De dunne: de schoorsteen van het ketelhuis, zo'n 50 meter hoog, de oudste van de twee. Het ketelhuis zelf is gesloopt; het nieuwe gebouw ernaast is erop geïnspireerd. De fabriek zette haar schoorstenen in haar logo.",
        "more": "https://www.boei.nl/projecten/schoorstenen-enka-ede",
    },
    {"name": "Langzaamverkeerbrug", "since": 2022, "at": (175295, 448655), "kind": "Fiets- en voetgangersbrug", "text": "De brug voor fietsers en voetgangers over het spoor, tussen de Enka-wijk en de Sysselt."},
    {"name": "De padden van het Enka-terrein", "since": 2019, "at": (175126, 448100), "kind": "Informatiebord"},
    {"name": "Monument voor de bombardementsslachtoffers", "at": (173869, 448765), "kind": "Oorlogsmonument"},
    {"name": "Oorlogsmonument", "at": (174956, 450248), "kind": "Oorlogsmonument"},
    {
        "name": "Mausoleum verzetstrijders",
        "at": (174999, 450636),
        "kind": "Erebegraafplaats · 1947",
        "text": "Op de Paasberg, waar de Duitsers eind 1944 bijna alle bomen kapten en loopgraven groeven, ligt de erebegraafplaats van het Edese verzet. Kort na de bevrijding stelden Derk Wildeboer en Zwerus de Nooij sr. een gedenkteken voor; de inwoners brachten er ruim 71.000 gulden voor bijeen. Landschapsarchitect Jan Bijhouwer ontwierp het, met J. Brands en A. ten Ham, en Canadese militairen die nog in Ede lagen hielpen bouwen. Prins Bernhard onthulde het op 2 augustus 1947. Op de bakstenen wand staan veertig natuurstenen platen met de namen in goud, en erboven: GESTEUND DOOR HET RECHT STREDEN ZIJ VOOR DE VRIJHEID. Bovenop staat een bronzen beeld van Piet Esser, een gevallen strijder die opstaat. Er liggen dertig verzetsstrijders begraven; elk jaar op 4 mei is hier de dodenherdenking van Ede.",
        "more": "https://nl.wikipedia.org/wiki/Mausoleum_(Ede)",
    },
    {"name": "Anna Maria Moens", "at": (174894, 450607), "kind": "Monument"},
    {"name": "Wilmina Lantaarn", "at": (174318, 450446), "kind": "Monument"},
    {"name": "Beeld: De Danser", "at": (174542, 448576), "kind": "Beeld"},
    {"name": "Beeld: Het Koningspaar", "at": (172890, 448367), "kind": "Kunstwerk"},
    {"name": "Beeld: Leeuw", "at": (175390, 449944), "kind": "Beeld"},
    {"name": "Beeld: De Grote Vlucht", "at": (174858, 450919), "kind": "Beeld"},
    {"name": "Beeld: Spelende beren", "at": (174485, 451001), "kind": "Beeld"},
    {"name": "Kinderboerderij De Proosdij", "at": (173196, 450270), "kind": "Kinderboerderij"},
]

# Series in the viewer's panel: a walk past these places, starting with the one nearest the Enka chimney.
CATEGORIES = [
    ("enka", "Enka-terrein", [
        "Spinnerijschoorsteen", "Ketelschoorsteen", "Het Ketelhuis (Stoer)", "Kantinegebouw", "Gymlokaal Enka", "Poortgebouw",
        "Westhal", "Hoek van het Carré", "Bitterzoutloods", "De padden van het Enka-terrein", "Langzaamverkeerbrug", "De Reehorst",
    ]),
    ("kazernes", "Kazernes", [
        "Stingerbol", "Johan Willem Friso kazerne", "Mauritskazerne", "Kantine van de kazerne", "Arthur Kool kazerne", "P.L. Bergansiuskazerne", "Van Essenkazerne",
        "Elias Beeckman kazerne", "Simon Stevinkazerne",
    ]),
    ("historisch", "Historische plekken", ["Station Ede Centrum", "Oude Kerk", "Sint Antonius van Padua", "Historisch Museum Ede", "Kasteel Hoekelum", "De Keetmolen"]),
    ("scholen", "Scholen", ["Koningin Beatrixschool", "Uniek", "De Franciscus"]),
    ("bedrijven", "Bedrijven en winkels", ["Riedel", "AH XL", "Stadspoort", "Pathé Ede"]),
    ("zorggeloof", "Zorg en geloof", ["Ziekenhuis Gelderse Vallei", "Ulu Moskee"]),
    ("oorlog", "Oorlog en herdenking", [
        "Monument voor de bombardementsslachtoffers", "Oorlogsmonument", "Mausoleum verzetstrijders",
    ]),
    ("kunst", "Kunst", [
        "Beeld: De Danser", "Beeld: Het Koningspaar", "Beeld: Leeuw", "Beeld: De Grote Vlucht", "Beeld: Spelende beren",
        "Anna Maria Moens", "Wilmina Lantaarn", "50PK",
    ]),
]

# What the viewer shows when a place is clicked; the rest of an entry is for the render.
INFO = ("note", "kind", "text", "address", "web", "more", "photo", "stop")

# Steel lattice masts that 3D BAG leaves out: none in Ede yet.
MASTS = []

# Monuments built here by hand: the Mausoleum on the Paasberg, from its outline in OpenStreetMap (way 734726076): a
# brick wall with the forty name plates, the bronze statue by Piet Esser on top and the flagpole beside it.
MONUMENTS = [
    {"kind": "mausoleum", "ring": [(174998.2, 450649.6), (174996.7, 450623.0), (175000.7, 450622.8), (175002.2, 450649.4)], "h": 4.2},
]

# Station roofs from the BGT "overkapping" that contains `at`: built after the AHN flight, so measured by eye.
# The roof of Ede-Wageningen is folded grey metal, a field of triangles with triangular skylights; the station hall at
# its south-east lobe has a flat green roof, and beside it stands a tall slender tower.
STATIONS = [{"name": "Ede-Wageningen", "since": 2023, "at": (174691, 448810), "height": 9.0, "fold": 1.4, "cell": 6.0,
             "grid": 18.0, "skylights": 0.14, "roof": (112, 116, 122), "glass": (176, 200, 214), "edge": (150, 154, 158),
             "columns": (230, 230, 226), "hall": (174680, 448720, 174770, 448792), "hall_height": 7.0,
             "hall_wall": (70, 74, 80), "hall_roof": (104, 128, 76),
             "tower": {"at": (174681, 448752), "h": 28.0, "w": 4.0}, "tower_colour": (206, 210, 212)}]

# Buildings that only need another colour than the renderer would pick, by address.
STYLED = [
    {"since": 2020, "addrs": ["Krimperij 3"], "style": "house", "wall": (236, 234, 226), "frame": (60, 62, 66)},
    # The muurwoningen behind the old north wall of the Carré: their outside is the factory facade.
    {"since": 2022, "name": "Muurwoningen Carré", "near": [(175010.6, 448626.8), (175226.3, 448556.4)], "built_from": 2010,
     "style": "factory", "wall": (118, 66, 52), "roof": (92, 94, 98), "frame": (232, 232, 226)},
    {"since": 2022, "name": "Muurwoningen Carré west", "near": [(174970.8, 448520.2), (174987.0, 448572.6), (174994.9, 448598.0)], "built_from": 2010,
     "style": "factory", "wall": (118, 66, 52), "roof": (92, 94, 98), "frame": (232, 232, 226)},
]

# The north wall of the Carré, which stood free between the demolition and the muurwoningen (AM, Op Enka: the photo
# of 2010): a brick wall two storeys high with empty window openings, with the arch to the Krimperij in it.
WALLS = [{"until": 2022, "path": [(175010.6, 448626.8), (175226.3, 448556.4)], "h": 10.5, "thick": 0.6,
          "gaps": [(175115, 448590, 7.0)], "wall": (128, 70, 54)},
         # The west wall, from the south up to the Poortgebouw (pointed out by the user).
         {"until": 2022, "path": [(174970.8, 448520.2), (174987.0, 448572.6), (174994.9, 448598.0)], "h": 10.5, "thick": 0.6,
          "wall": (128, 70, 54)}]

# Buildings so new that 3D BAG has no model of them yet, and AHN was flown before: extruded from their BGT footprint
# (by BAG id) to an estimated height.
NEW_BUILDINGS = [
    {"name": "P+R-garage", "since": 2026, "bag": "0228100000090180", "h": 12.0, "style": "parking",
     "wall": (150, 152, 150), "roof": (110, 112, 114)},
]

# Brick chimneys that 3D BAG leaves out, measured in AHN: height above the ground, radius at the foot and the top.
CHIMNEYS = [{"at": (175119.4, 448434.5), "height": 49.6, "r0": 2.6, "r1": 1.9, "wall": (150, 84, 62)}]

# The bridge for cyclists and walkers over the railway, along the Sysseltsepad in OpenStreetMap: from the ground on
# the Sysselt side up to the span over the tracks, and down the hairpin ramp on the Enka side. `span` is the stretch
# (indices into `path`) at full height; the ramps slope from there to the ground at both ends. The render is flat,
# so the deck clears the tracks by its full height.
BRIDGES = [
    {
        "name": "Langzaamverkeerbrug",
        "since": 2022,
        "path": [
            (175167.7, 448771.2), (175185.4, 448766.8), (175199.1, 448762.3), (175217.1, 448753.8), (175233.3, 448744.4),
            (175243.8, 448736.2), (175251.7, 448730.2), (175276.2, 448706.1), (175282.0, 448698.2), (175285.6, 448690.2),
            (175289.0, 448679.4), (175301.7, 448630.6), (175304.3, 448626.7), (175308.0, 448624.0), (175312.7, 448622.6),
            (175319.9, 448620.4), (175382.9, 448604.0), (175385.4, 448602.6), (175387.2, 448600.4), (175387.6, 448598.2),
            (175387.5, 448595.3), (175386.9, 448591.9), (175385.0, 448589.3), (175382.1, 448587.5), (175376.4, 448587.1),
            (175315.5, 448598.7), (175313.1, 448599.3), (175279.9, 448607.2), (175223.0, 448625.2),
        ],
        "span": (10, 11),
        "embankment": "end",  # the ramp on the Enka side (after the span) is a raised bank of earth, not on columns
        "height": 7.5,
        "width": 4.6,
        "deck": (212, 118, 60),  # the orange of the cycle path
        "wall": (200, 198, 190),
        "rail": (70, 74, 80),
    },
]

# Ground that changed after the BGT and the aerial photo: the strip between the railway and the Enka neighbourhood is
# a park now. Inside `area`, the BGT's sand, yards and the old asphalt road become grass, and the cycle paths (from the
# BGT and OpenStreetMap) the orange of the new cycle path.
GROUND = [
    {
        "name": "Park langs het spoor",
        "since": 2023,
        "area": [
            (175007.5, 448702.5), (175407.5, 448602.5), (175392.5, 448582.0), (175320.0, 448582.0), (175312.0, 448548.0),
            (175236.0, 448548.0), (175234.0, 448598.0), (175050.0, 448605.0), (175002.0, 448632.0),
        ],
        "cycle": (212, 118, 60),
    },
]

def circle(c, r, n=16):
    return [(round(c[0] + r * np.cos(2 * np.pi * i / n), 2), round(c[1] + r * np.sin(2 * np.pi * i / n), 2)) for i in range(n)]


# Things to play on (scripts/build_buildings.py add_props): a swing set across its circle, and the obstacle course.
PROPS = [
    {"kind": "swings", "since": 2020, "at": (175108.6, 448413.3), "angle": 18.0},
    # The loading dock on the south side of the Bitterzoutloods, where the trains stopped: platform and canopy stick
    # out, the doors lie back. From and to along the south facade (3D BAG), building on the left.
    {"kind": "dock", "since": 1919, "from": (174803.2, 448407.0), "to": (174878.4, 448382.3), "deck": 1.0, "depth": 2.6,
     "roof": 4.6, "overhang": 1.6, "bay": 5.5},
    {"kind": "obstacles", "since": 2020, "at": (175089.7, 448413.9), "r": 7.5},
]

# Ground that lies lower than its surroundings (scripts/build_buildings.py walls, render/scene.js lowers the ground
# there): the sunken schoolyard of De Ontdekking west of the old canteen, traced in AHN, 1.3 m down with a ramp at the
# west end.
SUNKEN = [
    {
        "name": "Schoolplein De Ontdekking", "since": 2021, "depth": 1.3,
        "area": [(175000.8, 448440.0), (174989.8, 448440.0), (174988.5, 448441.2), (174994.5, 448459.8), (174996.5, 448462.8),
                 (174970.2, 448471.5), (174966.0, 448474.8), (174967.8, 448478.0), (174982.2, 448473.5), (174984.8, 448477.5),
                 (174988.0, 448477.2), (174992.8, 448490.5), (175007.2, 448486.0), (175009.8, 448483.5), (175013.8, 448485.0),
                 (175016.0, 448483.2), (175004.0, 448446.2)],
    },
]

# Where children play, in RD meters: the schoolyard west of the old canteen and the terrace of the after-school care
# on its top floor (3D BAG has that roof at 7.3 m).
PLAY = [
    {"name": "Schoolplein", "since": 2021, "area": [(174992.5, 448489.0), (175014.0, 448482.5), (175008.0, 448462.0), (174995.0, 448465.0), (174988.0, 448477.0)], "h": -1.3, "kids": 16},
    # The playground on the lawn by the Ketelhuis, in two sand circles (OpenStreetMap): swings in the small one, an
    # obstacle course in the big one.
    {"name": "Speeltuin: schommels", "since": 2020, "area": circle((175108.6, 448413.3), 5.1), "h": 0.0, "kids": 3,
     "swing": {"at": (175108.6, 448413.3), "angle": 18.0}},
    {"name": "Speeltuin: hindernisbaan", "since": 2020, "area": circle((175089.7, 448413.9), 8.9), "h": 0.0, "kids": 6},
    # The schoolyard of the Koningin Beatrixschool, on the paved yard (BGT) east of the building in the Maandereng.
    {"name": "Schoolplein Maandereng", "since": 1985, "area": [(173345.5, 447822.0), (173356.0, 447821.0), (173353.0, 447846.0), (173345.0, 447851.0), (173340.0, 447846.0), (173343.0, 447830.0)], "h": 0.0, "kids": 14},
    {"name": "Terras van de BSO", "since": 2021, "area": [(175012.5, 448480.6), (175016.4, 448479.6), (175005.0, 448432.4), (175001.4, 448433.3)], "h": 7.3, "kids": 6},
]


# Bridges and viaducts from OpenStreetMap (bridge=*, layer >= 1): their deck height, how far the road climbs to it on
# either side, and how wide the deck is, by kind of road.
AUTO_DECK = 6.5
AUTO_RAMP = {"motorway": 130, "motorway_link": 90, "trunk": 110, "primary": 90, "secondary": 80}
AUTO_WIDTH = {"motorway": 13, "motorway_link": 7, "trunk": 11, "primary": 10, "secondary": 9, "tertiary": 9,
              "unclassified": 7, "residential": 7, "service": 5, "cycleway": 4, "footway": 3, "path": 3}


def auto_bridges(manual):
    """A deck with a ramp up and down for every road bridge in OpenStreetMap that the hand-made BRIDGES don't cover:
    the bridge itself at AUTO_DECK meters, the road climbing to it along the ways it continues in."""
    from shapely.geometry import LineString, Point

    from era import underpass_ids

    path = RAW / "osm_roads.json"
    if not path.exists():
        return []
    ways = [w for w in json.loads(path.read_text())["elements"] if w["type"] == "way" and w.get("geometry")]
    key = lambda g: (round(g["lat"], 7), round(g["lon"], 7))
    ends = {}
    for w in ways:
        for g in (w["geometry"][0], w["geometry"][-1]):
            ends.setdefault(key(g), []).append(w)
    rd = lambda w: [wgs_to_rd(g["lat"], g["lon"]) for g in w["geometry"]]
    # Covered by a hand-made bridge of any era: one that is not there yet (the Langzaamverkeerbrug) is not built from OSM either.
    taken = [LineString(b["path"]) for b in list(manual) + BRIDGES]

    def ramp(w, end, length):
        """Points along the ways that continue from `end` of w (0 start, -1 end), outward, `length` meters long."""
        pts, here, prev, left = [], w["geometry"][end], w, length
        while left > 0:
            nxt = [v for v in ends.get(key(here), []) if v is not prev and v["tags"].get("highway") == w["tags"]["highway"]]
            if not nxt:
                break
            v = nxt[0]
            g = v["geometry"] if key(v["geometry"][0]) == key(here) else v["geometry"][::-1]
            for a, b in zip(g, g[1:]):
                pa, pb = np.array(wgs_to_rd(a["lat"], a["lon"])), np.array(wgs_to_rd(b["lat"], b["lon"]))
                d = np.linalg.norm(pb - pa)
                if d >= left:
                    pts.append(tuple(pa + (pb - pa) * left / d))
                    left = 0
                    break
                pts.append(tuple(pb))
                left -= d
            here, prev = g[-1], v
        return pts

    out = []
    for w in ways:
        t = w["tags"]
        hw = t.get("highway", "").removesuffix("_link") if t.get("highway") != "motorway_link" else "motorway_link"
        if t.get("bridge") in (None, "no") or int(t.get("layer", "1") or 1) < 1 or hw not in AUTO_WIDTH:
            continue
        if w["id"] in underpass_ids():  # tagged as a bridge but it runs under the railway: the rail is the bridge
            continue
        span = rd(w)
        if any(LineString(span).distance(m) < 5 for m in taken):
            continue
        length = AUTO_RAMP.get(hw, 60)
        before, after = ramp(w, 0, length)[::-1], ramp(w, -1, length)
        pts = before + span + after
        out.append({"name": t.get("name") or t.get("ref") or f"brug {w['id']}", "path": [(round(x, 2), round(y, 2)) for x, y in pts],
                    "span": (len(before), len(before) + len(span) - 1), "height": AUTO_DECK, "width": AUTO_WIDTH[hw],
                    "deck": (98, 100, 106), "wall": (196, 194, 186), "rail": (150, 154, 158)})
    print(f"{len(out)} bridges from OpenStreetMap")
    return out


def era_skipped(name):
    """Whether a place is left out of the era being built, so its series can do without it."""
    return any(e["name"] == name and not in_era(e) for e in BUILDINGS + POINTS)


def bridge_heights(b):
    """The deck height at every point of a bridge's path: full over the span, sloping evenly to the ground at both ends."""
    p = np.asarray(b["path"], dtype=np.float64)
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s0, s1, h = cum[b["span"][0]], cum[b["span"][1]], b["height"]
    up = cum / s0 if s0 > 0 else np.ones_like(cum)  # no ramp on a side: at full height from its end
    down = (cum[-1] - cum) / (cum[-1] - s1) if cum[-1] > s1 else np.ones_like(cum)
    return [round(float(v), 2) for v in h * np.clip(np.minimum(up, down), 0, 1)]


def point_in_ring(x, y, ring):
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, np.roll(ring, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


ADDRESSES = None


def address_xy(name):
    """RD position of a BAG address such as "Geldersestraat 3" (data/raw/addresses.json, from scripts/fetch.py)."""
    global ADDRESSES
    if ADDRESSES is None:
        ADDRESSES = {label.split(", ")[0]: (x, y) for x, y, label in json.loads((RAW / "addresses.json").read_text())}
    return ADDRESSES[name]


def building_at(rings, x, y):
    for rid, ring, a in rings:
        if point_in_ring(x, y, ring):
            return rid, a
    return None, None


def main():
    feats = [json.loads(line) for line in open(RAW / "3dbag.jsonl")]
    rings = []
    for f in feats:
        t = f["transform"]
        V = np.array(f["vertices"], dtype=np.float64) * t["scale"] + t["translate"]
        co = f["CityObjects"][f["id"]]
        rings.append((f["id"], V[co["geometry"][0]["boundaries"][0][0]][:, :2], co["attributes"]))

    out = {"buildings": {}, "places": [], "masts": [], "hotspots": [], "categories": []}
    for b in filter(in_era, BUILDINGS):
        if "addrs" in b or "ats" in b:
            pts = b.get("ats") or [address_xy(a) for a in b["addrs"]]
            pids = [building_at(rings, *pt)[0] for pt in pts]
            assert all(pids), f"{b['name']}: no building at {b.get('addrs') or b['ats']}"
            x, y = np.mean(pts, axis=0)
            out["places"].append({"name": b["name"], "note": b.get("note"), "x": x, "y": y, "h": 4.0})
            out["hotspots"].append({"name": b["name"], "pids": pids, "x": x, "y": y, **{k: b[k] for k in INFO if k in b}})
            print(f"{b['name']}: {', '.join(pids)}")
            continue
        x, y = b["at"] if "at" in b else wgs_to_rd(*b["latlon"])
        pid, attrs = None, None
        for rid, ring, a in rings:
            if point_in_ring(x, y, ring):
                pid, attrs = rid, a
                break
        if pid is None:
            d = [(np.hypot(*(ring.mean(0) - (x, y))), rid, a) for rid, ring, a in rings]
            dist, pid, attrs = min(d, key=lambda t: t[0])
            if dist > 20:
                pid = None
        height = ((attrs or {}).get("b3_h_dak_max") or 0) - ((attrs or {}).get("b3_h_maaiveld") or 0)
        if "tower" in b:
            height = b["tower"]["top"]
        elif "turret" in b:
            height = b["turret"]["top"]
        out["places"].append({"name": b["name"], "note": b.get("note"), "x": x, "y": y, "h": round(height, 1)})
        if pid and "style" in b:
            out["buildings"][pid] = {k: v for k, v in b.items() if k not in ("at", "latlon", *INFO)}
        if pid:
            out["hotspots"].append({"name": b["name"], "pid": pid, "x": x, "y": y, **{k: b[k] for k in INFO if k in b}})
        print(f"{b['name']}: {pid or 'no building'}")
    for p in filter(in_era, POINTS):
        x, y = p["at"] if "at" in p else address_xy(p["addr"]) if "addr" in p else wgs_to_rd(*p["latlon"])
        out["places"].append({"name": p["name"], "note": p.get("note"), "x": x, "y": y, "h": 2.0})
        out["hotspots"].append({"name": p["name"], "x": x, "y": y, **{k: p[k] for k in INFO if k in p}})
    for b in filter(in_era, STYLED):
        if "near" in b:
            # Every building of `built_from` or later within 4 m of the line.
            from shapely.geometry import LineString, Polygon

            line = LineString(b["near"])
            pids = [rid for rid, ring, a in rings
                    if (a.get("oorspronkelijkbouwjaar") or 0) >= b["built_from"] and Polygon(ring).distance(line) < 4]
            name = b["name"]
        else:
            pids = {building_at(rings, *address_xy(a))[0] for a in b["addrs"]}
            name = ", ".join(b["addrs"])
        for pid in pids:
            assert pid, f"no building for {name}"
            out["buildings"][pid] = {"name": name, **{k: v for k, v in b.items() if k not in ("addrs", "near", "built_from", "name")}}
        print(f"{name}: restyled {len(pids)}")
    out["walls"] = list(filter(in_era, WALLS))
    out["new_buildings"] = list(filter(in_era, NEW_BUILDINGS))
    out["chimneys"] = [{**{k: v for k, v in c.items() if k != "at"}, "x": c["at"][0], "y": c["at"][1]} for c in filter(in_era, CHIMNEYS)]
    out["stations"] = list(filter(in_era, STATIONS))
    bridges = list(filter(in_era, BRIDGES))
    bridges += auto_bridges(bridges)
    out["bridges"] = [{**b, "heights": bridge_heights(b)} for b in bridges]
    out["ground"], out["play"] = list(filter(in_era, GROUND)), list(filter(in_era, PLAY))
    out["props"] = list(filter(in_era, PROPS))
    out["monuments"] = list(filter(in_era, MONUMENTS))
    out["sunken"] = list(filter(in_era, SUNKEN))
    for m in MASTS:
        x, y = wgs_to_rd(*m["latlon"])
        out["masts"].append({**{k: v for k, v in m.items() if k != "latlon"}, "x": x, "y": y})
    names = {h["name"] for h in out["hotspots"]}
    for key, label, members in CATEGORIES:
        members = [n for n in members if n in names or not era_skipped(n)]
        missing = [n for n in members if n not in names]
        assert not missing, f"{label}: no hotspot for {missing}"
        out["categories"].append({"key": key, "label": label, "names": members})
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "landmarks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
