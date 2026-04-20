"""
config.py — Lexicons, thresholds, and constants.

Lexicons are the *explanation layer*: they highlight which words
drove the score but do not determine it alone.  Context-aware
scoring in scorer.py downweights hits inside quotes, negated
phrases, and attribution clauses.
"""
from __future__ import annotations

APP_VERSION  = "3.0"
CSV_FILENAME = "biaslab_sessions.csv"

# ── Lexicons ──────────────────────────────────────────────────────

SENSATIONAL_WORDS = [
    "shocking", "shocked", "devastating", "devastated", "explosive",
    "bombshell", "outrageous", "outraged", "disgusting", "horrifying",
    "horrific", "terrifying", "catastrophe", "catastrophic", "crisis",
    "chaos", "chaotic", "furious", "enraged", "tragic", "tragedy",
    "nightmare", "meltdown", "scandal", "scandalous", "disaster",
    "disastrous", "stunning", "jaw-dropping", "unbelievable",
    "unthinkable", "alarming", "apocalyptic", "unprecedented",
    "staggering", "horrendous", "heartbreaking", "gut-wrenching",
    "ferocious", "blistering", "damning", "scathing", "seethe",
    "seething", "visionary", "sweeping", "transformative",
    "groundbreaking", "dominant", "remarkable", "overwhelming",
    "striking", "invaluable", "defining", "bold", "decisive",
    "irreversible", "unstoppable", "unwavering", "groundswell",
    "monumental", "towering", "unmatched", "unparalleled", "seismic",
    "extraordinary", "phenomenal", "turning point", "new era",
    "landmark", "renewed commitment", "national awakening",
    "moving forward", "ushering in", "new chapter", "defining moment",
    "bold vision", "strong leadership", "decisive leadership",
    "clear direction", "renewed strength", "renewed confidence",
    "tangible progress", "tangible benefits", "real results",
    "real progress", "meaningful change", "meaningful progress",
    "long overdue",
]

ABSOLUTE_WORDS = [
    "obviously", "clearly", "undeniably", "without question",
    "everyone knows", "nobody denies", "of course", "naturally",
    "evidently", "surely", "certainly", "undoubtedly", "definitely",
    "plainly", "absolutely", "needless to say", "make no mistake",
    "it is clear", "it's clear", "it goes without saying",
    "beyond doubt", "there is no question", "anyone can see",
    "any reasonable person", "the fact is", "the simple truth",
    "no one in their right mind", "is no exaggeration",
    "never been clearer", "there can be no doubt", "cannot be denied",
    "stands alone", "the truth is", "what cannot be ignored",
    "has succeeded in", "exactly what", "stands out for",
    "speaks for itself", "history will remember", "history will judge",
    "unmistakably", "unambiguously",
]

HEDGE_WORDS = [
    "may", "might", "could", "appears", "appeared", "seems", "seemed",
    "likely", "possibly", "perhaps", "apparently", "suggests",
    "indicates", "roughly", "approximately", "nearly", "in part",
    "to some extent", "is believed to", "preliminary", "tentative",
]

POSITIVE_WORDS = [
    "success", "successful", "triumph", "triumphant", "victory",
    "breakthrough", "hopeful", "hope", "thriving", "brilliant",
    "excellent", "wonderful", "great", "positive", "progress",
    "achievement", "innovative", "inspiring", "praise", "praised",
    "celebrated", "landmark", "historic", "milestone", "boost",
    "soar", "soared", "surge", "surged", "rally", "gains",
    "welcomed", "applauded", "prosperity", "prosperous", "confidence",
    "optimism", "optimistic", "pride", "proud", "ambition",
    "admiration", "foresight", "renewal", "resurgent", "flourish",
    "empowered", "encouraging", "promising", "widely praised",
    "strong leadership", "decisive leadership", "real results",
    "tangible benefits", "meaningful change", "dedicated",
    "commitment", "steady progress", "advancing", "forward-looking",
    "strength", "strengthen", "strengthened", "revitalize",
    "revitalized", "revival", "remarkable progress",
    "transformational", "inspirational", "bold reform", "pragmatic",
]

NEGATIVE_WORDS = [
    "failure", "failed", "fail", "defeat", "defeated", "plunge",
    "plunged", "crumble", "broken", "dangerous", "danger", "threat",
    "threatening", "fear", "feared", "worry", "worried", "concern",
    "concerned", "worst", "terrible", "awful", "weak", "weakened",
    "damage", "damaged", "harm", "hurt", "collapse", "collapsed",
    "ruined", "ruin", "criticized", "condemn", "condemned", "slump",
    "slumped", "decline", "declining", "plummet", "plummeted",
    "blasted", "scorned", "rejected", "pessimism", "stagnation",
    "stagnant", "mediocre", "obstruction", "division", "divisive",
    "hesitant", "disconnect", "incompetent", "incompetence",
    "dysfunction", "dysfunctional", "decay",
]

LEFT_LOADED = [
    "progressive", "equity", "marginalized", "systemic", "far-right",
    "far right", "white supremacist", "xenophobic", "climate crisis",
    "reproductive rights", "wealth gap", "corporate greed",
    "income inequality", "living wage", "climate denier",
    "structural racism", "late-stage capitalism", "dog whistle",
]

RIGHT_LOADED = [
    "woke", "liberal elite", "mainstream media", "illegal aliens",
    "radical left", "socialist", "communist", "globalist", "patriot",
    "patriotic", "traditional values", "law and order", "entitlement",
    "welfare state", "big government", "open borders",
    "cancel culture", "lamestream", "family values", "silent majority",
    "real americans", "career politician", "deep state",
]

FRAMING_WORDS = [
    "slammed", "attacked", "blasted", "ripped", "lashed out",
    "tore into", "savaged", "pounced", "enemy", "enemies", "traitor",
    "hero", "heroic", "coward", "regime", "crackdown", "mob",
    "radicals", "thugs", "puppet", "witch hunt", "clash", "clashed",
    "showdown", "stormed", "out-of-touch", "out of touch",
    "politically motivated", "disconnected from", "peddling",
    "smear", "smeared", "desperate attempt", "failed to grasp",
    "cling to", "hollow rhetoric", "empty rhetoric",
    "vocal minority", "stuck in the past", "refuses to adapt",
    "reflexive opposition", "knee-jerk opposition",
    "politically convenient",
]

SUPERLATIVES = [
    "worst", "best", "greatest", "biggest", "largest", "smallest",
    "unprecedented", "historic", "record-breaking", "record-high",
    "record-low", "first-ever", "never before", "most",
    "all-time low", "all-time high", "biggest ever", "largest ever",
    "once in a generation", "unmatched in history",
]

CITATION_SIGNALS = [
    "according to", "said", "stated", "reported", "noted",
    "announced", "confirmed", "told reporters", "press release",
    "in a statement", "testified", "wrote in", "explained",
    "acknowledged", "admitted", "documents show", "court records",
]

WEASEL_SIGNALS = [
    "sources say", "sources said", "experts say", "experts believe",
    "critics claim", "many believe", "some argue", "reportedly",
    "allegedly", "it is believed", "people are saying", "insiders say",
    "observers note", "anonymous sources", "familiar with the matter",
    "some say", "most agree", "few would deny", "widely regarded",
    "widely acknowledged", "officials familiar with", "those close to",
    "those familiar with", "people familiar with", "industry sources",
    "several analysts", "some analysts", "some have suggested",
    "has been described as", "described as", "said to be",
    "is understood to", "it is thought", "commentators say",
    "stakeholders say", "officials say", "insiders said",
    "experts are calling", "senior official", "a senior official",
    "business leaders", "community leaders", "policy experts",
    "what many see as", "what many describe as", "widely seen as",
    "widely viewed as", "growing consensus", "broad consensus",
    "public sentiment",
]

# ── Negation & attribution ────────────────────────────────────────

NEGATION_WORDS = [
    "not", "no", "never", "neither", "nor", "n't", "cannot",
    "without", "hardly", "barely", "scarcely", "rarely",
    "nothing", "nobody", "none",
]
NEGATION_WINDOW = 3  # words after negation to flag

ATTRIBUTION_VERBS = [
    "said", "stated", "reported", "noted", "announced", "confirmed",
    "told", "explained", "acknowledged", "admitted", "testified",
    "wrote", "writes", "claimed", "argued", "maintained", "insisted",
    "declared", "asserted", "remarked", "commented", "responded",
    "added", "observed", "suggested", "cautioned", "warned",
    "emphasized", "stressed",
]

# ── Scoring half-points (hyperbolic curve tuning) ─────────────────

HALF_SENSATIONAL = 10.0
HALF_ABSOLUTE    =  7.0
HALF_FRAMING     = 10.0
HALF_SLANT       = 18.0
HALF_WEASEL      =  8.0
HALF_EXCLAIM     =  5.0
HALF_CAPS        =  3.0
HALF_SUPERLATIVE =  6.0
HALF_CLICKBAIT   = 40.0
HALF_CONFIDENCE  = 200.0

# ── Context weights ──────────────────────────────────────────────

WEIGHT_AUTHOR_VOICE  = 1.0   # normal author prose
WEIGHT_ATTRIBUTED    = 0.30  # "X said ..." clauses
WEIGHT_QUOTED        = 0.15  # inside direct quotes
WEIGHT_NEGATED       = 0.20  # negated lexicon hits

# ── UI labels & verdict thresholds ───────────────────────────────

RADAR_LABELS = [
    "Loaded\nLanguage", "Sentiment\nImbalance", "Subjectivity",
    "Source\nOpacity", "Sensational\nFraming", "Political\nSlant",
]

VERDICT_THRESHOLDS = [
    (20,  "MINIMAL BIAS",  "Reads like fairly neutral reporting."),
    (40,  "MILD BIAS",     "Mostly neutral with some loaded phrasing."),
    (60,  "MODERATE BIAS", "Notable slant — read a second source."),
    (80,  "STRONG BIAS",   "Heavy slant — treat as opinion, not reporting."),
    (101, "EXTREME BIAS",  "Reads like an opinion piece or propaganda."),
]

CONFIDENCE_ZONES = [
    (0.40, "LOW",    "Low confidence — insufficient evidence"),
    (0.70, "MEDIUM", "Moderate confidence — needs review"),
    (1.01, "HIGH",   "Strong evidential support"),
]

# ── Demo articles ────────────────────────────────────────────────

SAMPLE_TITLE = (
    "Radical Politicians Slam Patriotic Plan In Shocking Meltdown "
    "As Critics Seethe"
)
SAMPLE_BODY = (
    "In a truly shocking and devastating turn of events, the radical "
    "left lashed out at a patriotic new plan that traditional values "
    "voters have been celebrating for weeks. Sources say the outrageous "
    "response was a catastrophe waiting to happen, while insiders say "
    "the scandal is only beginning.\n\n"
    "Experts believe the liberal elite mainstream media will obviously "
    "try to bury this explosive story, but make no mistake — the "
    "American people are furious. Critics claim the plan is dangerous, "
    "but many believe those critics are simply socialist globalists "
    "pushing a woke agenda. Clearly, the far-right is nothing like what "
    "the systemic critics pretend.\n\n"
    "Reportedly, the welfare state crowd has already begun a meltdown, "
    "slamming anyone who disagrees as an enemy of progress. Undoubtedly, "
    "this scandal will define the coming weeks as patriots and thugs "
    "clash in an unbelievable nationwide crackdown.\n\n"
    "A small number of analysts may point out that the proposal "
    "actually includes modest tax incentives, a review mechanism and a "
    "two-year sunset clause. Those details, however, are largely ignored "
    "by commentators focused on the political showdown. Anonymous "
    "sources familiar with the matter say the outcome is all but certain, "
    "and observers note the party leadership has already begun preparing "
    "talking points for the next news cycle.\n\n"
    "The bill is scheduled for a committee hearing on Thursday. The "
    "sponsor's office declined repeated requests for comment, though a "
    "brief press release confirmed that a formal statement will be "
    "released after the session. In the meantime, it goes without "
    "saying that the stakes could not be higher, and anyone can see "
    "where this story is heading.\n\n"
    "Meanwhile, a record-breaking number of constituents have called in "
    "to express their views, with the absolute worst outcome — a full "
    "stalemate — looking more likely by the hour. Naturally, the "
    "president's office has declined comment, even as the biggest ever "
    "lobbying campaign gears up behind the scenes. Historic precedents "
    "suggest the final vote could go either way, but critics are already "
    "warning of unprecedented consequences."
)

SAMPLE_NEUTRAL_TITLE = "Senate Committee Advances Infrastructure Bill in 14-8 Vote"
SAMPLE_NEUTRAL_BODY = (
    "The Senate Environment and Public Works Committee voted 14 to 8 on "
    "Wednesday to advance a bipartisan infrastructure bill that would "
    "allocate $303 billion for roads, bridges and transit systems over "
    "the next five years.\n\n"
    'Senator Tom Carper, the committee chair, said the bill "reflects '
    'months of careful negotiation" and called the vote "an important '
    'step forward." Senator Shelley Moore Capito, the ranking Republican '
    "member, voted in favor but noted that several provisions still "
    "need revision before a full Senate vote.\n\n"
    "The bill includes $110 billion for roads and bridges, $66 billion "
    "for passenger and freight rail, $49 billion for public transit, and "
    "$25 billion for airports. An additional $55 billion is set aside for "
    "clean water infrastructure, according to a summary released by the "
    "committee.\n\n"
    "The Congressional Budget Office has not yet released a full cost "
    "estimate. Committee staff said the spending would be partially "
    "offset by repurposing unspent pandemic relief funds and projected "
    "economic growth, though independent analysts have questioned whether "
    "those offsets are sufficient.\n\n"
    "A floor vote is expected within three to four weeks. The White House "
    "issued a statement supporting the bill but did not comment on "
    "specific provisions."
)
