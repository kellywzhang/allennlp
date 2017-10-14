# coding: utf8
from __future__ import unicode_literals

from spacy.symbols import *


EXC = {}

#EXCLUDE_EXC = ["Ill", "ill", "Its", "its", "Hell", "hell", "Shell", "shell",
#               "Shed", "shed", "were", "Were", "Well", "well", "Whore", "whore"]

# Times

for h in range(1, 12 + 1):
    hour = str(h)

    for period in ["a.m.", "am"]:
        EXC[hour + period] = [
            {ORTH: hour},
            {ORTH: period, LEMMA: "a.m."}
        ]
    for period in ["p.m.", "pm"]:
        EXC[hour + period] = [
            {ORTH: hour},
            {ORTH: period, LEMMA: "p.m."}
        ]


# Abbreviations

ABBREVIATIONS = {
    "Mt.": [
        {ORTH: "Mt.", LEMMA: "Mount"}
    ],

    "Ak.": [
        {ORTH: "Ak.", LEMMA: "Alaska"}
    ],

    "Ala.": [
        {ORTH: "Ala.", LEMMA: "Alabama"}
    ],

    "Apr.": [
        {ORTH: "Apr.", LEMMA: "April"}
    ],

    "Ariz.": [
        {ORTH: "Ariz.", LEMMA: "Arizona"}
    ],

    "Ark.": [
        {ORTH: "Ark.", LEMMA: "Arkansas"}
    ],

    "Aug.": [
        {ORTH: "Aug.", LEMMA: "August"}
    ],

    "Calif.": [
        {ORTH: "Calif.", LEMMA: "California"}
    ],

    "Colo.": [
        {ORTH: "Colo.", LEMMA: "Colorado"}
    ],

    "Conn.": [
        {ORTH: "Conn.", LEMMA: "Connecticut"}
    ],

    "Dec.": [
        {ORTH: "Dec.", LEMMA: "December"}
    ],

    "Del.": [
        {ORTH: "Del.", LEMMA: "Delaware"}
    ],

    "Feb.": [
        {ORTH: "Feb.", LEMMA: "February"}
    ],

    "Fla.": [
        {ORTH: "Fla.", LEMMA: "Florida"}
    ],

    "Ga.": [
        {ORTH: "Ga.", LEMMA: "Georgia"}
    ],

    "Ia.": [
        {ORTH: "Ia.", LEMMA: "Iowa"}
    ],

    "Id.": [
        {ORTH: "Id.", LEMMA: "Idaho"}
    ],

    "Ill.": [
        {ORTH: "Ill.", LEMMA: "Illinois"}
    ],

    "Ind.": [
        {ORTH: "Ind.", LEMMA: "Indiana"}
    ],

    "Jan.": [
        {ORTH: "Jan.", LEMMA: "January"}
    ],

    "Jul.": [
        {ORTH: "Jul.", LEMMA: "July"}
    ],

    "Jun.": [
        {ORTH: "Jun.", LEMMA: "June"}
    ],

    "Kan.": [
        {ORTH: "Kan.", LEMMA: "Kansas"}
    ],

    "Kans.": [
        {ORTH: "Kans.", LEMMA: "Kansas"}
    ],

    "Ky.": [
        {ORTH: "Ky.", LEMMA: "Kentucky"}
    ],

    "La.": [
        {ORTH: "La.", LEMMA: "Louisiana"}
    ],

    "Mar.": [
        {ORTH: "Mar.", LEMMA: "March"}
    ],

    "Mass.": [
        {ORTH: "Mass.", LEMMA: "Massachusetts"}
    ],

    "May.": [
        {ORTH: "May.", LEMMA: "May"}
    ],

    "Mich.": [
        {ORTH: "Mich.", LEMMA: "Michigan"}
    ],

    "Minn.": [
        {ORTH: "Minn.", LEMMA: "Minnesota"}
    ],

    "Miss.": [
        {ORTH: "Miss.", LEMMA: "Mississippi"}
    ],

    "N.C.": [
        {ORTH: "N.C.", LEMMA: "North Carolina"}
    ],

    "N.D.": [
        {ORTH: "N.D.", LEMMA: "North Dakota"}
    ],

    "N.H.": [
        {ORTH: "N.H.", LEMMA: "New Hampshire"}
    ],

    "N.J.": [
        {ORTH: "N.J.", LEMMA: "New Jersey"}
    ],

    "N.M.": [
        {ORTH: "N.M.", LEMMA: "New Mexico"}
    ],

    "N.Y.": [
        {ORTH: "N.Y.", LEMMA: "New York"}
    ],

    "Neb.": [
        {ORTH: "Neb.", LEMMA: "Nebraska"}
    ],

    "Nebr.": [
        {ORTH: "Nebr.", LEMMA: "Nebraska"}
    ],

    "Nev.": [
        {ORTH: "Nev.", LEMMA: "Nevada"}
    ],

    "Nov.": [
        {ORTH: "Nov.", LEMMA: "November"}
    ],

    "Oct.": [
        {ORTH: "Oct.", LEMMA: "October"}
    ],

    "Okla.": [
        {ORTH: "Okla.", LEMMA: "Oklahoma"}
    ],

    "Ore.": [
        {ORTH: "Ore.", LEMMA: "Oregon"}
    ],

    "Pa.": [
        {ORTH: "Pa.", LEMMA: "Pennsylvania"}
    ],

    "S.C.": [
        {ORTH: "S.C.", LEMMA: "South Carolina"}
    ],

    "Sep.": [
        {ORTH: "Sep.", LEMMA: "September"}
    ],

    "Sept.": [
        {ORTH: "Sept.", LEMMA: "September"}
    ],

    "Tenn.": [
        {ORTH: "Tenn.", LEMMA: "Tennessee"}
    ],

    "Va.": [
        {ORTH: "Va.", LEMMA: "Virginia"}
    ],

    "Wash.": [
        {ORTH: "Wash.", LEMMA: "Washington"}
    ],

    "Wis.": [
        {ORTH: "Wis.", LEMMA: "Wisconsin"}
    ]
}


TOKENIZER_EXCEPTIONS = dict(EXC)
TOKENIZER_EXCEPTIONS.update(ABBREVIATIONS)


# Remove EXCLUDE_EXC if in exceptions
"""
for string in EXCLUDE_EXC:
    if string in TOKENIZER_EXCEPTIONS:
        TOKENIZER_EXCEPTIONS.pop(string)
"""

# Abbreviations with only one ORTH token

ORTH_ONLY = [
    "'d",
    "a.m.",
    "Adm.",
    "Bros.",
    "co.",
    "Co.",
    "Corp.",
    "D.C.",
    "Dr.",
    "e.g.",
    "E.g.",
    "E.G.",
    "Gen.",
    "Gov.",
    "i.e.",
    "I.e.",
    "I.E.",
    "Inc.",
    "Jr.",
    "Ltd.",
    "Md.",
    "Messrs.",
    "Mo.",
    "Mont.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "p.m.",
    "Ph.D.",
    "Rep.",
    "Rev.",
    "Sen.",
    "St.",
    "vs.",
]

