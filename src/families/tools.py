"""Shared tools for all architectural families.

All families use the same tools for fair comparison.
"""

WIKI_PARAGRAPHS = {
    "scott derrickson": "Scott Derrickson (born July 16, 1966) is an American filmmaker. He is best known for directing Doctor Strange (2016) and The Exorcism of Emily Rose (2005).",
    "ed wood": "Edward Davis Wood Jr. (October 10, 1924 - December 10, 1978) was an American filmmaker, actor, and author.",
    "nora batty": "Nora Batty is a fictional character in the BBC sitcom Last of the Summer Wine, portrayed by Kathy Staff from 1973 to 2010.",
    "kathy staff": "Kathy Staff (born Minnie Higginbottom; 12 July 1928 - 13 December 2008) was a British actress. She served as a councillor on Dukinfield Town Council.",
    "boston celtics": "The Boston Celtics are an American professional basketball team based in Boston. They play their home games at TD Garden.",
    "td garden": "TD Garden (originally FleetCenter) is a multi-purpose arena in Boston, Massachusetts. It opened on September 30, 1995.",
    "arthur's magazine": "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia. It was founded by Timothy Shay Arthur.",
    "first for women": "First for Women is a woman's magazine published by Bauer Media Group. The magazine was started in 1989.",
    "jaws": "Jaws is a 1975 American thriller film directed by Steven Spielberg. It is based on the 1974 novel by Peter Benchley.",
    "steven spielberg": "Steven Allan Spielberg (born December 18, 1946) is an American filmmaker. He is one of the most commercially successful directors in history.",
    "france": "France is a country in Western Europe. Its capital is Paris. It is known for its wine, cuisine, and the Eiffel Tower.",
    "paris": "Paris is the capital and most populous city of France, with a population of over 2.1 million residents.",
    "python": "Python is a high-level programming language created by Guido van Rossum and first released in 1991.",
    "albert einstein": "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist who developed the theory of relativity.",
    "moscow state university": "Moscow State University (MSU) was founded in 1755 by Mikhail Lomonosov. It is the oldest university in Russia.",
    "harvard university": "Harvard University is a private Ivy League research university in Cambridge, Massachusetts, established in 1636.",
    "world war ii": "World War II was a global conflict that lasted from 1939 to 1945, involving most of the world's nations.",
    "united states": "The United States of America is a country primarily located in North America. Its capital is Washington, D.C.",
    "united kingdom": "The United Kingdom is a country in Northwestern Europe, comprising England, Scotland, Wales, and Northern Ireland.",
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
