import re
import numpy as np

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