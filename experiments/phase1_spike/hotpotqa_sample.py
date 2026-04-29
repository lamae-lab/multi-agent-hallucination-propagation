"""Shared HotpotQA sample questions and mock Wikipedia search tool.

Used by all spike scripts to ensure identical test inputs across frameworks.
"""

QUESTIONS = [
    {
        "id": "q1",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "expected_answer": "Yes",
        "supporting_facts": ["Scott Derrickson", "Ed Wood"],
    },
    {
        "id": "q2",
        "question": "What government position was held by the woman who portrayed Nora Batty in Last of the Summer Wine?",
        "expected_answer": "Councillor",
        "supporting_facts": ["Nora Batty", "Kathy Staff"],
    },
    {
        "id": "q3",
        "question": "The arena where the Weights Celtics play opened in what year?",
        "expected_answer": "1999",
        "supporting_facts": ["Boston Celtics", "TD Garden"],
    },
    {
        "id": "q4",
        "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
        "expected_answer": "Arthur's Magazine",
        "supporting_facts": ["Arthur's Magazine", "First for Women"],
    },
    {
        "id": "q5",
        "question": "What nationality is the director of the film Jaws?",
        "expected_answer": "American",
        "supporting_facts": ["Jaws (film)", "Steven Spielberg"],
    },
]

# Mock Wikipedia paragraphs keyed by entity name
WIKI_PARAGRAPHS = {
    "scott derrickson": "Scott Derrickson (born July 16, 1966) is an American filmmaker. He is best known for directing Doctor Strange (2016) and The Exorcism of Emily Rose (2005).",
    "ed wood": "Edward Davis Wood Jr. (October 10, 1924 - December 10, 1978) was an American filmmaker, actor, and author. He is often cited as one of the worst film directors of all time.",
    "nora batty": "Nora Batty is a fictional character in the BBC sitcom Last of the Summer Wine, portrayed by Kathy Staff from 1973 to 2010.",
    "kathy staff": "Kathy Staff (born Minnie Higginbottom; 12 July 1928 - 13 December 2008) was a British actress. She served as a councillor on Dukinfield Town Council.",
    "boston celtics": "The Boston Celtics are an American professional basketball team based in Boston. They play their home games at TD Garden.",
    "td garden": "TD Garden (originally FleetCenter) is a multi-purpose arena in Boston, Massachusetts. It opened on September 30, 1995. The current naming rights deal began in 1999.",
    "arthur's magazine": "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia. It was founded by Timothy Shay Arthur.",
    "first for women": "First for Women is a woman's magazine published by Bauer Media Group. The magazine was started in 1989.",
    "jaws (film)": "Jaws is a 1975 American thriller film directed by Steven Spielberg. It is based on the 1974 novel by Peter Benchley.",
    "steven spielberg": "Steven Allan Spielberg (born December 18, 1946) is an American filmmaker. He is one of the most commercially successful directors in history.",
}


def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic.

    Args:
        query: The topic to search for.

    Returns:
        A paragraph of information about the topic, or a not-found message.
    """
    key = query.lower().strip()
    for entity, paragraph in WIKI_PARAGRAPHS.items():
        if entity in key or key in entity:
            return paragraph
    return f"No Wikipedia article found for '{query}'."
