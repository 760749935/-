import csv
data = [
    # Fake entries (8)
    {"post_id": "VovinPrgel", "post_text": "refugees_1", "user_id": "", "username": "", "image_id": "refugees_1",
     "timestamp": "Tue Jan 12 20:08:27 +0000 2016", "label": "fake"},
    {"post_id": "643498101165137921", "post_text": "Willkommen..!! http://t.co/yLvLNMr1pW", "user_id": "563819167",
     "username": "FabriBerta", "image_id": "refugees_1", "timestamp": "Mon Sep 14 18:54:37 +0000 2015",
     "label": "fake"},
    {"post_id": "691324169728892928", "post_text": "Bienvenue en Allemagne  https://t.co/GEYeww6gz5",
     "user_id": "1461552590", "username": "83Django", "image_id": "refugees_1",
     "timestamp": "Sun Jan 24 18:18:20 +0000 2016", "label": "fake"},
    {"post_id": "688780462203006976",
     "post_text": "RT @Megafauna2: @AmyMek @anonym_exmuslim @creepingsharia @barenakedislam @iammrsc THIS is what happens when U let in 2 many Muslims! https:...",
     "user_id": "2990770013", "username": "PatColina", "image_id": "refugees_1",
     "timestamp": "Sun Jan 17 17:50:33 +0000 2016", "label": "fake"},
    {"post_id": "317788846932770816",
     "post_text": "@IEarthPictures: Unbelievable Shot Of The 2012 Supermoon In Rio de Janeiro. http://t.co/0JIGcCL3q0 where the heck was I for this?",
     "user_id": "203986866", "username": "gabbydaviss", "image_id": "rio_moon_1",
     "timestamp": "Sat Mar 30 00:02:19 +0000 2013", "label": "fake"},
    {"post_id": "725705340067573760",
     "post_text": "Dude I love to snowboard but this girl is my hero dude didn't even know about the bear... crazy https://t.co/OnfU8uTfjI",
     "user_id": "363716074", "username": "cidvicious6969", "image_id": "snowboard_girl_1",
     "timestamp": "Thu Apr 28 15:16:50 +0000 2016", "label": "fake"},
    {"post_id": "443148818226610176",
     "post_text": "ביזה באוקראינה\nThe unknown soldier steal geese from Crimean Tatars #Ukraine #Crimea http://t.co/LKmgorTLuQ",
     "user_id": "82405831", "username": "ZEATU", "image_id": "soldier_stealing_1",
     "timestamp": "Mon Mar 10 22:18:06 +0000 2014", "label": "fake"},
    {"post_id": "739719450048352256",
     "post_text": "Little Syrian girl sells chewing gum on the street so she can feed herself... https://t.co/pGhagPBxHb",
     "user_id": "17517730", "username": "cjflines", "image_id": "syrian_children_1",
     "timestamp": "Mon Jun 06 07:23:54 +0000 2016", "label": "fake"},

    # Real entries (8)
    {"post_id": "648058517430104064",
     "post_text": "#refugees wait to enter a refugee camp in Gevgelija #Macedonia. For @nytimes @nytimesphoto  https://t.co/sczah03cMb http://t.co/XeX2aKLUdh",
     "user_id": "422494755", "username": "Samuel_Aranda_", "image_id": "refugees_10",
     "timestamp": "Sun Sep 27 08:56:05 +0000 2015", "label": "real"},
    {"post_id": "650974702194835456",
     "post_text": "RT @HumzaYousaf: For those think refugees come Europe for life of luxury -does this look like luxury? We wouldn't last a day in camps http:...",
     "user_id": "65211831", "username": "_AbuShayma", "image_id": "refugees_11",
     "timestamp": "Mon Oct 05 10:03:58 +0000 2015", "label": "real"},
    {"post_id": "652419526630490112",
     "post_text": "RT @BabarBloch: Hegyeshalom, #Hungary: Refugees long walk to freedom. #Europe provides sanctuary http://t.co/b093HjKA4A",
     "user_id": "819259538", "username": "gabrielaleu", "image_id": "refugees_12",
     "timestamp": "Fri Oct 09 09:45:11 +0000 2015", "label": "real"},
    {"post_id": "688028684071088129",
     "post_text": "RT @Refugees: Today Mediterranean sea arrivals to Europe reach 1 million. 49% are from Syria.  https://t.co/nAUZy25H5X https://t.co/RgJ89LH...",
     "user_id": "3164342562", "username": "ShreySakhuja", "image_id": "refugees_13",
     "timestamp": "Fri Jan 15 16:03:15 +0000 2016", "label": "real"},
    {"post_id": "686161466408435712",
     "post_text": "Titta på barnen. De satt närmare 30 barn i den här båten. Barn som flyr i en överfull båt över havet. https://t.co/YtTfJxXh6d",
     "user_id": "348374251", "username": "julianfirpo", "image_id": "refugees_14",
     "timestamp": "Sun Jan 10 12:23:36 +0000 2016", "label": "real"},
    {"post_id": "692336769086398464",
     "post_text": "RT @Refugees: Vulnerable refugees + migrants are in need of protection + assistance. Our strategy: https://t.co/3PNso9q8nT #Europe https://...",
     "user_id": "2875267084", "username": "philomenanalty", "image_id": "refugees_15",
     "timestamp": "Wed Jan 27 13:22:03 +0000 2016", "label": "real"},
    {"post_id": "692395995313917952",
     "post_text": "RT @UNRefugeeAgency: These Photos Show Another Face Of The #RefugeeCrisis: https://t.co/SDiePXmOnT #supportrefugees #refugees https://t.co/...",
     "user_id": "155909582", "username": "BabarBloch", "image_id": "refugees_16",
     "timestamp": "Wed Jan 27 17:17:24 +0000 2016", "label": "real"},
    {"post_id": "715839940521824256", "post_text": "godless people of Syria https://t.co/yhByc45Lgt",
     "user_id": "3275569286", "username": "BloodMoonEufra8", "image_id": "syrian_children_2",
     "timestamp": "Fri Apr 01 09:55:16 +0000 2016", "label": "real"}
]

# Write to CSV
with open('social_media_posts.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['post_id', 'post_text', 'user_id', 'username', 'image_id', 'timestamp', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in data:
        writer.writerow(row)

print("CSV file created successfully with 8 real and 8 fake entries.")