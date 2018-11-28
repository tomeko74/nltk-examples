import nltk
"""
from nltk.book import *
from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import nps_chat
from nltk.corpus import brown
from nltk.corpus import reuters
"""
from nltk.corpus import inaugural

"""
print("\n-----------------------------------------------\n")
print(f"\nNazwa Text 1 to: {text1}\n")

print("\n-----------------------------------------------\n")
print("Kontekst wystąpienia słowa 'monstrous' w text1\n")
text1.concordance("monstrous")

print("\n-----------------------------------------------\n")
print("Wyrazy występujące w podobnym kontekście co 'monstrous', czyli podobne:\n")
print("w text1:\n")
text1.similar("monstrous")
print("\nw text2:\n")
text2.similar("monstrous")

print("\n-----------------------------------------------\n")
print("Współdzielony kontekst słów 'monstrous' i 'very'\n")
text2.common_contexts(["monstrous", "very"])

print("\n-----------------------------------------------\n")
print("Wykres pozycji występowania poszczególnych słów w text4 (sklejka przemów prezydentów):\n")
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America", "liberty", "constitution"])

print("\n-----------------------------------------------\n")
print("Wygenerowany losowo tekst (100 słów) w stylu text3 (losowo powybierane frazy w stylu):\n")
print(f"\nChyba nie działa w NLTK 3:\n{text3.generate(100)}\n")

print("\n-----------------------------------------------\n")
print("Liczba słów i znaków przestankowych w text3:\n")
print(len(text3))

print("\n-----------------------------------------------\n")
print("Posortowana lista unikalnych słów i znaków przestankowych w text3:\n")
print(sorted(set(text3)))
print("\nLiczba unikalnych słów i znaków przestankowych w text3:\n")
print(len(sorted(set(text3))))

print("\n-----------------------------------------------\n")
print("Jak często średnio używane jest każde słowo w text3:\n")
print(f"{len(text3)/len(set(text3))}\n")

print("\n-----------------------------------------------\n")
print("Ile razy występuje słowo 'smote' w text3:\n")
print(text3.count("smote"))

print("\n-----------------------------------------------\n")
print("Ile % stanowi słowo 'a' w text4:\n")
print(100*text4.count("a")/len(text4))

print("\n-----------------------------------------------\n")
print("Lista słów dłuższych niż 15 liter w text1:\n")
V = set(text1)
long_words = [w for w in V if len(w) > 15]
print(sorted(long_words))

print("\n-----------------------------------------------\n")
print("Lista słów dłuższych niż 7 liter, które występują więcej niż 7 razy w text5:\n")
fdist5 = FreqDist(text5)
print(sorted([w for w in set(text5) if len(w) > 7 and fdist5[w] > 7]))

print("\n-----------------------------------------------\n")
print("Stworzenie bigramów z listy słów:\n")
print(list(bigrams(['more', 'is', 'said', 'than', 'done'])))

print("\n-----------------------------------------------\n")
print("Najczęstsze pary słów (kolokacje) w text4:\n")
print(text4.collocations())

print("\n-----------------------------------------------\n")
print("Rozkład częstości długości słów (jak długie są najpopularniejsze):\n")
fdist = FreqDist([len(w) for w in text1])
print(f"{list(fdist.keys())}\n")
print(f"\n(długość, liczność):\n{fdist.items()}\n")
print(f"najczęstsza długość: {fdist.max()}\n")
print(f"ile % stanowią słowa 3-literowe: {fdist.freq(3)}\n")

#różne nieprzydatne przykłady w pythonie
#.......

# ......
print("Różne chatboty zaimplementowane w nltk:\n")
nltk.chat.chatbots()


print("\n-----------------------------------------------\n")
print("Jakie mamy korpusy językowe:\n")

print(gutenberg.fileids())

print("\nIle wyrazów ma 'austen-emma.txt'?\n")
emma = gutenberg.words('austen-emma.txt')
print(len(emma))
print("\nKonkordancje 'surprize' w 'austen-emma.txt'?\n")
emma = nltk.Text(gutenberg.words('austen-emma.txt'))
print(emma.concordance("surprize"))

print("\nStatystyki korpusów językowych:\n")
print("chars/words\twords/sents\twords/vocab\tfileid")
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
    print(f"{int(num_chars/num_words)}\t\t{int(num_words/num_sents)}\t\t{int(num_words/num_vocab)}\t\t{fileid}")

print("\nInne korpusy:\n")
print("korpus webowy:\n")
for fileid in webtext.fileids():
    print(fileid, webtext.raw(fileid)[:65], '...')

print("\nkorpus czatowy:\n")
chatroom = nps_chat.posts('10-19-20s_706posts.xml')
print(chatroom[123])

print("\nBrown korpus (pierwszy milion słów elektronicznie z 1961 roku):\n")
print("\nKategorie:\n")
print(brown.categories())

print("\nKategoria 'news':\n")
print(brown.words(categories='news'))

print("\nKonkretny dokument:\n")
print(brown.words(fileids=['cg22']))

print("\nZdania z kategorii 'news', 'editorial', 'reviews':\n")
print(brown.sents(categories=['news', 'editorial', 'reviews']))

print("\nStatystyka występowania wyrazów modalnych w kategorii 'news':\n")
news_text = brown.words(categories='news')
fdist = nltk.FreqDist([w.lower() for w in news_text])
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals:
    print(m + ':', fdist[m],)

print("\nStatystyka występowania wyrazów modalnych w różnych kategoriach:\n")
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))

genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions=genres, samples=modals)

print("\nTeraz korpus Reutersa:\n")
print("\nNazwy plików:\n")
print(reuters.fileids())
print("\nKategorie:\n")
print(reuters.categories())
print("\nKategorie dla podanego pliku:\n")
print(reuters.categories('training/9865'))
print("\nKategorie dla podanych plików:\n")
print(reuters.categories(['training/9865', 'training/9880']))
print("\nPliki dla podanej kategorii:\n")
print(reuters.fileids('barley'))
print("\nPliki dla podanych kategorii:\n")
print(reuters.fileids(['barley', 'corn']))
print("\nSłowa dla podanego pliku 'training/9865' (pierwsze 14):\n")
print(reuters.words('training/9865')[:14])
print("\nSłowa dla podanych plików 'training/9865', 'training/9880':\n")
print(reuters.words(['training/9865', 'training/9880']))
print("\nSłowa dla podanej kategorii 'barley':\n")
print(reuters.words(categories='barley'))
print("\nSłowa dla podanych kategorii 'barley', 'corn':\n")
print(reuters.words(categories=['barley', 'corn']))

print("\nTeraz korpus mów wstępnych prezydentów:\n")
print("\nNazwy plików:\n")
print(inaugural.fileids())
print("\nWycięte same lata z plików:\n")
print([fileid[:4] for fileid in inaugural.fileids()])

print("\nJak słowa 'america' i 'citizen' były używane przez lata:\n")
cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target))
cfd.plot()

print("\nKorpusy w innych językach:\n")
print("\nHiszpański:\n")
print(nltk.corpus.cess_esp.words())
print("\nKolejny hiszpański:\n")
print(nltk.corpus.floresta.words())
print("\nHinduski:\n")
print(nltk.corpus.indian.words('hindi.pos'))
print("\nDeklaracja Praw Człowieka w ponad 300 językach:\n")
print(nltk.corpus.udhr.fileids())
print("\nSłowa Deklaracji po jawajsku:\n")
print(nltk.corpus.udhr.words('Javanese-Latin1')[11:])

print("\nRozkład długości słów w poszczególnych językach w Deklaracji:\n")
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in nltk.corpus.udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)
"""


