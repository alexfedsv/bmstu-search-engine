from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc,
    NewsNERTagger,
    PER,
    LOC,
    ORG,
    DatesExtractor,
    MoneyExtractor,
    AddrExtractor,
    NamesExtractor
)

text = ''

with open('article.txt', 'r') as file:
    for line in file:
        text += line
file.close()


segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)
dates_extractor = DatesExtractor(morph=morph_vocab)
doc = Doc(text)
doc.segment(segmenter)
# print('--------------------------------------------')
# print('Segmentation')
# print('слова:')
# print('--------------------------------------------')
# print(doc.tokens[:])
# print('--------------------------------------------')
# print('Segmentation')
# print('предложения')
# print('--------------------------------------------')
# print(doc.sents[:])
# print('--------------------------------------------')
# print('Morphology')
# print('--------------------------------------------')
# doc.tag_morph(morph_tagger)
# print(doc.tokens[:])
# doc.sents[0].morph.print()
# print('--------------------------------------------')
# print('Lemmatization')
# print('--------------------------------------------')
# for token in doc.tokens:
    # token.lemmatize(morph_vocab)

# print(doc.tokens[:])
# t = {_.text: _.lemma for _ in doc.tokens }
# print(t)
# print('--------------------------------------------')
# print('Syntax')
# print('--------------------------------------------')
# doc.parse_syntax(syntax_parser)
# doc.sents[0].syntax.print()
# print('!--------------------------------------------!')
# print('!--------------------------------------------!')
# print('!--------------------------------------------!')
print('NER')
print('--------------------------------------------')
doc.tag_ner(ner_tagger)
#print(doc.spans[:])
doc.ner.print()


for span in doc.spans:
    span.normalize(morph_vocab)
names = {_.text: _.normal for _ in doc.spans}
print(names)

for span in doc.spans:
    if span.type == PER:
        span.extract_fact(names_extractor)

names = {_.normal: _.fact.as_dict for _ in doc.spans if _.fact}
print(names)

import re

# Паттерн для извлечения дат в формате "DD.MM.YYYY"
patterns = [
    r"\bXIX век(?:а|)|\b\d{4} год(?:а|)|\b\d{4}—\d{4} годах|\b\d{4}-х|\b\d{4}—\d{4}|(?:\bXIX век(?:а|)|\b\d{4} году)\b"
]
# Находим все даты в тексте
dates = []
for pattern in patterns:
    dates.extend(re.findall(pattern, text))

# Выводим найденные даты
for date in dates:
    print(date)
print('Создание собственных правил:')
from yargy import Parser, rule
from yargy.predicates import gram, dictionary
print('---------R_1----------')
R_1 = rule(gram('ADJF'), dictionary({'символике', 'языки', 'цифр', 'букв', 'символов', 'системами'}))
parser = Parser(R_1)
for match in parser.findall(text):
    print([x.value for x in match.tokens])
print('---------R_2----------')
R_2 = rule(gram('VERB'), dictionary({'Домене', 'аббат'}))
parser = Parser(R_2)
for match in parser.findall(text):
    print([x.value for x in match.tokens])
print('---------R_3----------')
R_3 = rule(dictionary({'Домене', 'аббат'}), gram('VERB'))
parser = Parser(R_3)
for match in parser.findall(text):
    print([x.value for x in match.tokens])
print('---------R_4----------')
R_4 = rule(gram('ADJF').optional(), gram('NOUN'), gram('CONJ'), gram('ADJF').optional(), gram('NOUN'))
parser = Parser(R_4)
for match in parser.findall(text):
    print([x.value for x in match.tokens])
