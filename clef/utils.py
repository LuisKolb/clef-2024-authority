from typing import List, Tuple, Dict, TypedDict, Union, Optional, NamedTuple
import textwrap
import json 
import os
import re
import numpy as np

class RumorDict(TypedDict):
    id: str
    rumor: str
    label: str
    timeline: List[List[str]]
    evidence: List[List[str]]
    retrieved_evidence: List[List[Union[str, int, float]]]

class RankedDocs(NamedTuple):
    author_account: str
    authority_tweet_id:str
    doc_text: str
    rank: int
    score: float


def load_rumors_from_jsonl(filepath: Union[str, os.PathLike]) -> List[RumorDict]:
    jsons = []
    with open(filepath, encoding='utf8') as file:
        for line in file:
            jsons += [json.loads(line)]
    return jsons


def write_trec_format_output(filename: str, data: List[List[Union[str, int, float]]], tag: str) -> None:
    """
    Writes data to a file in the TREC format.

    Parameters:
    - filename (str): The name of the file to write to.
    - data (List[Tuple[str, int, int, float]]): A list of tuples, where each tuple contains:
        - rumor_id (str): The unique ID for the given rumor.
        - authority_tweet_id (int): The unique ID for the authority tweet.
        - rank (int): The rank of the authority tweet ID for that given rumor_id.
        - score (float): The score given by the model for the authority tweet ID.
    - tag (str): The string identifier of the team/model.
    """
    with open(filename, 'w') as file:
        for rumor_id, authority_tweet_id, rank, score in data:
            line = f"{rumor_id}\tQ0\t{authority_tweet_id}\t{rank}\t{score}\t{tag}\n"
            file.write(line)


def combine_rumors_with_trec_file_judgements(jsons, trec_judgements_path):
    """
    create a list of RankedDocs objects in key retrieved_evidence from TREC-formatted file

    Parameters:
        - jsons: list of json-like rumors from dataset
        - trec_judgements_path: filepath to TREC-formatted file containing rank and score

    Returns:
    list of json-like rumors from dataset, with the key retrieved_evidence populated with a list of RankedDocs-like objects
    """
    trec_by_id = {}

    with open(trec_judgements_path, 'r') as file:
        for line in file:
            rumor_id, _, evidence_id, rank, score, _ = line.split('\t')
            if rumor_id not in trec_by_id:
                trec_by_id[rumor_id] = {}
            trec_by_id[rumor_id][evidence_id] = (rank, score)

    for i, item in enumerate(jsons):
        item['retrieved_evidence'] = []

        doc_ranks = trec_by_id[item['id']]
        for author_account, tweet_id, tweet_text in item['timeline']:
            if tweet_id in doc_ranks:
                rank, score = doc_ranks[tweet_id]
                item['retrieved_evidence'] += [[
                    author_account,
                    tweet_id,
                    tweet_text,
                    int(rank),
                    float(score),
                ]]
        
        jsons[i] = item
    return jsons


def write_jsonlines_from_dicts(filename: Union[str, os.PathLike], dicts: List[Dict]) -> None:
    with open(filename, 'w') as file:
        for item in dicts:
            file.write(f'{json.dumps(item)}\n')


def clean_tweet(text):
    # source: https://github.com/Fatima-Haouari/AuFIN/blob/main/code/utils.py
    if (text is None) or (text is np.nan):
        return ""
    else:
        text = re.sub(r"http\S+", " ", text)  # remove urls
        text = re.sub(r"RT ", " ", text)  # remove rt
        text = re.sub(r"@[\w]*", " ", text)  # remove handles
        text = re.sub(r"[\.\,\#_\|\:\?\?\/\=]", " ", text)  # remove special characters
        text = re.sub(r"\t", " ", text)  # remove tabs
        text = re.sub(r"\n", " ", text)  # remove line jump
        text = re.sub(r"\s+", " ", text)  # remove extra white space
        text = re.sub(r"\u201c", " ", text)  # remove “ character
        text = re.sub(r"\u201d", " ", text)  # remove “ character
        # accents = re.compile(r'[\u064b-\u0652\u0640]') # harakaat and tatweel (kashida) to remove

        # arabic_punc= re.compile(r'[\u0621-\u063A\u0641-\u064A\d+]+') # Keep only Arabic letters/do not remove numbers
        # text=' '.join(arabic_punc.findall(accents.sub('',text)))
        text = text.strip()
        return text
    
def clean_tweet_aggressive(text):
    # source: https://github.com/Fatima-Haouari/AuFIN/blob/main/code/utils.py
    if (text is None) or (text is np.nan):
        return ""
    else:
        text = re.sub(r"http\S+", " ", text)  # remove urls
        text = re.sub(r"RT ", " ", text)  # remove rt
        text = re.sub(r"@[\w]*", " ", text)  # remove handles
        
        # TODO: does this improve results? hashtags are... whacky; see e.g. rumor AuRED_104
        text = re.sub(r"[\.\,\|\:\?\?\/\=]", " ", text)  # remove special characters

        text = re.sub(r"#[\w]*", " ", text)  # remove handles

        text = re.sub(r"\t", " ", text)  # remove tabs
        text = re.sub(r"\n", " ", text)  # remove line jump
        text = re.sub(r"\s+", " ", text)  # remove extra white space
        text = re.sub(r"\u201c", "", text)  # remove “ character
        text = re.sub(r"\u201d", "", text)  # remove ” character
        text = re.sub(r"\u2018", "", text)  # remove ‘ character
        text = re.sub(r"\u2019", "", text)  # remove ’ character
        text = re.sub(r'\"', " ", text)  # remove " character

        # see: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
        emojis = r"(?:[0-9#*]️⃣|[☝✊-✍🎅🏂🏇👂👃👆-👐👦👧👫-👭👲👴-👶👸👼💃💅💏💑💪🕴🕺🖐🖕🖖🙌🙏🛀🛌🤌🤏🤘-🤟🤰-🤴🤶🥷🦵🦶🦻🧒🧓🧕🫃-🫅🫰🫲-🫸][🏻-🏿]?|⛓(?:️‍💥)?|[⛹🏋🏌🕵](?:️‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|❤(?:️‍[🔥🩹])?|🇦[🇨-🇬🇮🇱🇲🇴🇶-🇺🇼🇽🇿]|🇧[🇦🇧🇩-🇯🇱-🇴🇶-🇹🇻🇼🇾🇿]|🇨[🇦🇨🇩🇫-🇮🇰-🇵🇷🇺-🇿]|🇩[🇪🇬🇯🇰🇲🇴🇿]|🇪[🇦🇨🇪🇬🇭🇷-🇺]|🇫[🇮-🇰🇲🇴🇷]|🇬[🇦🇧🇩-🇮🇱-🇳🇵-🇺🇼🇾]|🇭[🇰🇲🇳🇷🇹🇺]|🇮[🇨-🇪🇱-🇴🇶-🇹]|🇯[🇪🇲🇴🇵]|🇰[🇪🇬-🇮🇲🇳🇵🇷🇼🇾🇿]|🇱[🇦-🇨🇮🇰🇷-🇻🇾]|🇲[🇦🇨-🇭🇰-🇿]|🇳[🇦🇨🇪-🇬🇮🇱🇴🇵🇷🇺🇿]|🇴🇲|🇵[🇦🇪-🇭🇰-🇳🇷-🇹🇼🇾]|🇶🇦|🇷[🇪🇴🇸🇺🇼]|🇸[🇦-🇪🇬-🇴🇷-🇹🇻🇽-🇿]|🇹[🇦🇨🇩🇫-🇭🇯-🇴🇷🇹🇻🇼🇿]|🇺[🇦🇬🇲🇳🇸🇾🇿]|🇻[🇦🇨🇪🇬🇮🇳🇺]|🇼[🇫🇸]|🇽🇰|🇾[🇪🇹]|🇿[🇦🇲🇼]|🍄(?:‍🟫)?|🍋(?:‍🟩)?|[🏃🚶🧎](?:‍(?:[♀♂]️(?:‍➡️)?|➡️)|[🏻-🏿](?:‍(?:[♀♂]️(?:‍➡️)?|➡️))?)?|[🏄🏊👮👰👱👳👷💁💂💆💇🙅-🙇🙋🙍🙎🚣🚴🚵🤦🤵🤷-🤹🤽🤾🦸🦹🧍🧏🧔🧖-🧝](?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|🏳(?:️‍(?:⚧️|🌈))?|🏴(?:‍☠️|󠁧(?:󠁢(?:󠁥󠁮󠁧|󠁳󠁣󠁴)󠁿)?)?|🐈(?:‍⬛)?|🐕(?:‍🦺)?|🐦(?:‍[⬛🔥])?|🐻(?:‍❄️)?|👁(?:️‍🗨️)?|👨(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨|👦(?:‍👦)?|👧(?:‍[👦👧])?|[👨👩]‍(?:👦(?:‍👦)?|👧(?:‍[👦👧])?)|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳])|🏻(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏼-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏼(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏻🏽-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏽(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏻🏼🏾🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏾(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏻-🏽🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏿(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏻-🏾]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?)?|👩(?:‍(?:[⚕⚖✈]️|❤️‍(?:[👨👩]|💋‍[👨👩])|👦(?:‍👦)?|👧(?:‍[👦👧])?|👩‍(?:👦(?:‍👦)?|👧(?:‍[👦👧])?)|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳])|🏻(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏼-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏼(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏻🏽-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏽(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏻🏼🏾🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏾(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏻-🏽🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏿(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏻-🏾]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?)?|[👯🤼🧞🧟](?:‍[♀♂]️)?|😮(?:‍💨)?|😵(?:‍💫)?|😶(?:‍🌫️)?|🙂(?:‍[↔↕]️)?|🧑(?:‍(?:[⚕⚖✈]️|🤝‍🧑|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]|(?:🧑‍)?🧒(?:‍🧒)?)|🏻(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏼-🏿]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏼(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏻🏽-🏿]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏽(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏻🏼🏾🏿]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏾(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏻-🏽🏿]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏿(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏻-🏾]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?)?|[©®‼⁉™ℹ↔-↙↩↪⌚⌛⌨⏏⏩-⏳⏸-⏺Ⓜ▪▫▶◀◻-◾☀-☄☎☑☔☕☘☠☢☣☦☪☮☯☸-☺♀♂♈-♓♟♠♣♥♦♨♻♾♿⚒-⚗⚙⚛⚜⚠⚡⚧⚪⚫⚰⚱⚽⚾⛄⛅⛈⛎⛏⛑⛔⛩⛪⛰-⛵⛷⛸⛺⛽✂✅✈✉✏✒✔✖✝✡✨✳✴❄❇❌❎❓-❕❗❣➕-➗➡➰➿⤴⤵⬅-⬇⬛⬜⭐⭕〰〽㊗㊙🀄🃏🅰🅱🅾🅿🆎🆑-🆚🈁🈂🈚🈯🈲-🈺🉐🉑🌀-🌡🌤-🍃🍅-🍊🍌-🎄🎆-🎓🎖🎗🎙-🎛🎞-🏁🏅🏆🏈🏉🏍-🏰🏵🏷-🐇🐉-🐔🐖-🐥🐧-🐺🐼-👀👄👅👑-👥👪👹-👻👽-💀💄💈-💎💐💒-💩💫-📽📿-🔽🕉-🕎🕐-🕧🕯🕰🕳🕶-🕹🖇🖊-🖍🖤🖥🖨🖱🖲🖼🗂-🗄🗑-🗓🗜-🗞🗡🗣🗨🗯🗳🗺-😭😯-😴😷-🙁🙃🙄🙈-🙊🚀-🚢🚤-🚳🚷-🚿🛁-🛅🛋🛍-🛒🛕-🛗🛜-🛥🛩🛫🛬🛰🛳-🛼🟠-🟫🟰🤍🤎🤐-🤗🤠-🤥🤧-🤯🤺🤿-🥅🥇-🥶🥸-🦴🦷🦺🦼-🧌🧐🧠-🧿🩰-🩼🪀-🪈🪐-🪽🪿-🫂🫎-🫛🫠-🫨]|🫱(?:🏻(?:‍🫲[🏼-🏿])?|🏼(?:‍🫲[🏻🏽-🏿])?|🏽(?:‍🫲[🏻🏼🏾🏿])?|🏾(?:‍🫲[🏻-🏽🏿])?|🏿(?:‍🫲[🏻-🏾])?)?)+"
        text = re.sub(emojis, "", text)


        #unused
        # accents = re.compile(r'[\u064b-\u0652\u0640]') # harakaat and tatweel (kashida) to remove

        # arabic_punc= re.compile(r'[\u0621-\u063A\u0641-\u064A\d+]+') # Keep only Arabic letters/do not remove numbers
        # text=' '.join(arabic_punc.findall(accents.sub('',text)))
        text = text.strip()
        return text


def wrap(text: str):
    """
    Print strings line-wrapped.

    Parameters:
    - text (str): The text you want to print.
    """

    wrapped_text = textwrap.fill(text, width=80) #text is the object which you want to print
    print(wrapped_text)

