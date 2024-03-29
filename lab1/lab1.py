# -*- coding: utf-8 -*-

from razdel import sentenize
from razdel import tokenize as razdel_tokenize
from pymystem3 import Mystem
from pymorphy2 import MorphAnalyzer
morph = MorphAnalyzer()
mystem = Mystem()
'''
Вариант который правильно разбивается на два предложения:
The sentence is: Мы отправились в путешествие на машине 10 марта 2024 года. Там провели неделю, и, вернувшись домой, обнаружили запасной ключ в почтовом ящике.
Trying to break the sentence:
Substring(0, 58, 'Мы отправились в путешествие на машине 10 марта 2024 года.')
Substring(59, 142, 'Там провели неделю, и, вернувшись домой, обнаружили запасной ключ в почтовом ящике.')
Ниже вариант который обрабатывается не правильно:
The sentence is: Мы отправились в путешествие на машине 10 марта 2024 г. Там провели неделю, и, вернувшись домой, обнаружили запасной ключ в почтовом ящике.
Trying to break the sentence:
Substring(0, 139, 'Мы отправились в путешествие на машине 10 марта 2024 г. Там провели неделю, и, вернувшись домой, обнаружили запасной ключ в почтовом ящике.')
'''
text_to_break = '''Мы отправились в путешествие на машине 10 марта 2024 г. Там провели неделю, и, вернувшись домой, обнаружили запасной ключ в почтовом ящике.'''
print("------------------")
print("The sentence is:", text_to_break)
print("Trying to break the sentence:")
tokenized_break = list(sentenize(text_to_break))
for token in tokenized_break[:2]:
    print(token)
print("------------------")

text = """Конфедерати́вные Шта́ты Аме́рики, известные также как Конфедерати́вные Шта́ты, КША, Конфедера́ция или Юг (англ. The Confederate States of America, CSA, The Confederacy, the South), Южане — де-факто независимое государство (на протяжении 1862—1863 гг. почти добившееся признания своего суверенитета Британской империей и Францией, но после поражения в битве при Геттисберге так и не признанное ни одной страной), существовавшее в период с 1861 по 1865 год в южной части Северной Америки, на части территории современных Соединённых Штатов Америки. Конфедерация южных штатов образовалась в результате выхода (сецессии) 13 южных рабовладельческих штатов из состава Соединённых Штатов Америки. Конфедеративные Штаты были противником Соединённых Штатов во время Гражданской войны в США. Потерпев поражение в войне, Конфедеративные Штаты прекратили своё существование; штаты, их составлявшие, были захвачены вооружёнными силами Соединённых Штатов и реинтегрированы в США во время длительного процесса Реконструкции Юга.
Первое совещание сторонников сецессии состоялось 22 ноября 1860 года в городе Аббевилл (Южная Каролина). Конфедеративные Штаты Америки были образованы 4 февраля 1861 года шестью южными штатами (Южная Каролина, Миссисипи, Флорида, Алабама, Джорджия и Луизиана) после того, как были утверждены результаты выборов президента США, победителем которых был объявлен Авраам Линкольн (представитель северян, выступавший с позиций осуждения, ограничения и запрета распространения на новые территории рабства, которое являлось основой экономики южных штатов). Эти шесть южных штатов и присоединившийся к ним 2 марта Техас объявили о своём выходе из состава США и возвращении властям штатов полномочий, делегированных по Конституции 1787 года федеральному правительству. Среди прочего эти полномочия включали в себя контроль над военными укреплениями (фортами), портами и таможнями, расположенными на территории штатов, и сбор налогов и пошлин. Месяц спустя после образования КША, 4 марта, принял присягу 16-й президент Соединённых Штатов Америки Авраам Линкольн. В своей инаугурационной речи он назвал сецессию «юридически ничтожной» и объявил, что США не собираются вторгаться на территорию южных штатов, но готовы применить силу для сохранения своего контроля над федеральной собственностью и сбором налогов. 12 апреля 1861 года войска штата Южной Каролины под командованием генерала Пьера Г. Т. Борегара разбомбили стоявший в Чарльстонской гавани федеральный Форт-Самтер, принудив его гарнизон к капитуляции. Сражение за форт Самтер положило начало Гражданской войне. После бомбардировки Самтера Линкольн призвал штаты, оставшиеся в Союзе, предоставить ему солдат для силового восстановления контроля над Самтером и остальными южными фортами, защиты федеральной столицы и сохранения Союза. В ответ на президентское обращение ещё четыре южных штата (Виргиния, Арканзас, Теннесси и Северная Каролина) объявили о выходе из США и присоединении к Конфедерации. Штаты Кентукки и Миссури остались «пограничными штатами» в составе США, но какое-то время имели по два правительства, одно из которых поддерживало Союз, другое — Конфедерацию. Проконфедеративные правительства этих штатов включили подконтрольные им территории в состав Конфедерации, и это позволяет считать членами КША 13 штатов. Из территорий, тогда ещё не имевших прав штатов, прошение о вступлении в КША подали Аризона и Нью-Мексико. Также Конфедеративные Штаты были поддержаны пятью «цивилизованными» племенами с Индейской территории — чероки, чокто, чикасо, криков, семинолов. Не все американские рабовладельческие штаты присоединились к Конфедерации, этого не сделали штаты Мэриленд и Делавэр.
После четырёх лет Гражданской войны командующий Армией Северной Виргинии генерал Роберт Ли 9 апреля 1865 года в местечке Аппоматтокс, Виргиния, капитулировал перед главнокомандующим армией Союза генералом Улиссом С. Грантом. За 6 дней до этого — 3 апреля правительство КША покинуло Ричмонд и перебралось в городок Данвилл, Виргиния. Но уже через неделю, 10 апреля, члены правительства были вынуждены покинуть и его. Фактически в этот день Конфедеративные Штаты Америки прекратили своё существование. По иронии судьбы, последнее заседание кабинета министров Джефферсона Дэвиса, датированное 2 мая 1865 года, прошло там же, где состоялось первое совещание сторонников сецессии — в Аббевилле (Южная Каролина). Бывший президент Конфедерации Джефферсон Дэвис был задержан 10 мая и более года провёл в тюрьме. Позже он был обвинён в государственной измене, но вина так и не была доказана. С апреля по июнь капитулировали остальные армии Конфедерации. Последним, 6 ноября 1865 года, спустил флаг корабль «Шенандоа» («Shenandoah»).
Южные штаты ждал долгий и тяжёлый период «Реконструкции» и возвращения в состав США. Условием возвращения было принятие абсолютно новых конституций штатов, запрещающих рабство, и ратификация соответствующей поправки к Конституции США. Первым обратно был принят Теннесси (24 июня 1866), а последним — Джорджия (15 июля 1870).
"""
# print("By razdel")
"""tokenized = list(sentenize(text))
for token in tokenized[:100]:
    print(token)"""
print("------------------")
print("Токены (mystem):")
analysis = mystem.analyze(text)
print(analysis)
tokens_mystem = [item['text'] for item in analysis if 'text' in item]
print(tokens_mystem)
print("------------------")
print("Данные (razdel):")
substrings_razdel = list(razdel_tokenize(text))[:1000]
print(substrings_razdel)
print("---")
print("Токены (razdel):")
tokens_razdel = [substring.text for substring in substrings_razdel]
print(tokens_razdel)
print("------------------")
print("Леммы (mystem):")
lemma_mystem = mystem.lemmatize(text)
print(lemma_mystem)
print("------------------------------Токенизация Mystem vs razdel.tokenize-------------------------------------")
print("Токены (mystem):")
print(tokens_mystem)
print("Токены (razdel):")
print(tokens_razdel)
print("Очищенные токены:")
tokens_mystem_cleaned = [word.strip(' .,;:!?()[]') for word in tokens_mystem if word.isalpha()] # й как самостоятельный токен, 'Форт', 'Самтер' && 'форт', 'Самтер' учтены
tokens_razdel_cleaned = [word.strip(' .,;:!?()[]') for word in tokens_razdel if word.isalpha()] # 'форт', 'Самтер' учтен, а 'Форт', 'Самтер' нет
print(tokens_mystem_cleaned)
print(tokens_razdel_cleaned)
# razdel не оставлят пробелов в виде токенов
print("------------------------------Лемматизация Mystem vs Pymorphy-------------------------------------")
print("Леммы (mystem):")
lemma_mystem = mystem.lemmatize(' '.join(tokens_mystem_cleaned))
print(lemma_mystem)
print("------------------")
print("Леммы (morph):")
lemma_morph = [morph.parse(token) for token in tokens_mystem_cleaned]
# print(lemma_morph)
lemmas = [word[0].normal_form for word in lemma_morph]
print(lemmas)
# morph все переводит в нижний регистр




