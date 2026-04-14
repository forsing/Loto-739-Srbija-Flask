from app import create_app

# Ulazna tačka Flask aplikacije
app = create_app()

if __name__ == "__main__":
    app.run(debug=True)





"""
Inspiracija/Nadogradnja - Inspiration/Upgrade
https://github.com/themrandray/Lottelligence/tree/main



cd /Loto-739-Srbija-Flask-main
python3 run.py

Zatim u pregledaču otvori adresu koju Flask ispiše 
(obično http://127.0.0.1:5000/).




* Restarting with watchdog (fsevents)
 * Debugger is active!
 * Debugger PIN: 472-737-116
127.0.0.1 - - [14/Apr/2026 00:36:16] "POST /top-numbers-api HTTP/1.1" 200 -
127.0.0.1 - - [14/Apr/2026 00:38:34] "POST /run HTTP/1.1" 200 -
127.0.0.1 - - [14/Apr/2026 00:38:34] "GET /static/style.css HTTP/1.1" 304 -
127.0.0.1 - - [14/Apr/2026 00:38:57] "POST /top-combinations-api HTTP/1.1" 200 -





Rezultati
Predlog za sledeće izvlačenje (7 brojeva)
2 - x - 10 - y - 33 — z — 39
Pet uzastopnih izvlacenj u obelezjima, frekvencija i razmaci racunati od cele istorije u CSV; 
predlog = prosek tri modela (SGD, RF, XGB/GB). 
Informativno — nije obecanje ishoda.





Flask server

Flask app, app/routes.py, run.py


Loto 7/39

app/services/dataset.py 
— normalizacija za Num1-Num7, opcija CSV bez zaglavlja (7 kolona), 
opcija pripremljeni (date, n1-n7, draw_no). 
Unutrašnji format: n1-n7, opseg 1-39, 7 različitih brojeva po redu.

app/services/experiment.py 
— za loto739: max_num=39, k_main=7, u lag-u uključuje i n7 ako kolona postoji.

Dodato: 
izračunavanje predložene sledeće kombinacije iz poslednjeg izvlačenja 
(model: prev→curr, ulaz = poslednji žreb) i prikaz u šablonu.

Posle „Pokreni eksperiment“, iznad tabela sa metrikama pojavljuje se blok „Predlog za sledeće izvlačenje (7 brojeva)“ 
sa sedam brojeva (sortirano u kodu kao rastući niz, prikaz sa „ — “).

Kako se računa
Isti lag kao u eksperimentu: model uči prethodno izvlačenje → sledeće.

Za sledeći korak kao ulaz uzima se poslednje izvlačenje iz CSV-a (one-hot 39 pozicija).

Za predikciju se koristi isti tip modela koji je u tom pokretanju izabran kao najbolji 
po zbiru rangova (best_overall); ako nešto ne štima, koristi se slučajna šuma (RF).

Uzima se 7 brojeva sa najvećom predviđenom verovatnoćom.

Ispod bloka objašnjava se da je to informativno, ne garancija.

Ponovo pokreni python3 run.py, uradi Pokreni eksperiment i skroluj do Rezultati 
— predlog je odmah ispod naslova te sekcije.



df_norm = ceo CSV posle normalizacije (svi redovi / sva izvlačenja do kraja fajla).
predict_next_combo pravi lag nad celim tim podacima, pa X_train / Y_train obuhvataju 
sve parove prethodno → trenutno iz te istorije.
Ulaz za „sledeće“ je curr_vec poslednjeg reda = poslednja kombinacija u tom istom fajlu.
Za predlog se koristi ceo fajl sa svim do tada izvučenim kombinacijama.



SEED = 39 i random_state=39 u modelima da bi isti podaci davali istu predikciju.
SEED = 39 u modelima, set_reproducible_seed() na početku eksperimenta i predikcije, i n_jobs=1 kod RandomForest radi determinističkog paralelizma.
SEED = 39 u app/services/models.py 
— svi random_state sada koriste 39.

RandomForestClassifier i XGBClassifier: 
n_jobs=1 (sa -1 isti seed ponekad i dalje može da „pomera“ stabla između pokretanja).

_set_seed() na početku run_experiment i predict_next_combo: random.seed(39) i np.random.seed(39).

np.argsort(..., kind="stable") za izbor top-7 i u _hit_at_k — pri istim verovatnoćama redosled je uvek isti.

Uz iste podatke i isti izbor „najboljeg“ modela posle eksperimenta, 
predložena kombinacija treba da bude uvek ista pri svakom pokretanju. 
Potpuna identičnost između različitih mašina / verzija sklearn-a 
nije matematički garantovana zbog brojeva u pokretnom zarezu, 
ali u praksi na istom okruženju ovo je ono šta se traži.




Evo šta najviše donosi nadogradnja u ovakvom setupu (bez obećanja „boljeg lutrijskog ishoda“):

1.Bolji ulaz u model — umesto samo one-hot prethodnog žreba: kratka istorija (npr. 3-5 prethodnih kola), brojač pojavnosti u prozoru, „gap“ od poslednjeg izlaska broja, parovi/trojke kao agregati. Lutrija je i dalje slučajna, ali model dobija bogatiji signal za učenje strukture u podacima.

2.Kalibracija verovatnoća — predict_proba često nije dobro skaliran; posle treninga CalibratedClassifierCV ili Platt/Isotonic pomaže metrikama (LogLoss, Brier) i stabilnijem izboru top-7.

3.Ansambl predikcije za „sledeću“ kombinaciju — umesto samo „najboljeg“ modela po metrikama, usrediniti verovatnoće sa sva tri (ili težinski glas po validacionom skoru). Često smanjuje ekstremne greške jednog modela.

4.Validacija u vremenu — striktno vremenski split (šta već radi se) + eventualno walk-forward (klizeći prozor) da se vidi da li predlog za sledeće kolose stabilno ili se preprilagođava prošlost.

5.Jasno razdvojen trening za metrike vs. trening za finalni predlog — metrike na train/test splitu; za predlog sledećeg kola trenirati na celoj istoriji do danas (što već radi) — to je u redu, ne da isti split slučajno „curi“ u feature-e budućnosti (leakage).

6.Grananje pri istim verovatnoćama — već ima stable sort; ako često ima iste proba, razmotriti sekundarni kriterijum (npr. istorijska frekvencija broja) da se uvek dobije isti skup pri malim numeričkim razlikama.



Implementirajući bogatije obelježja (više prethodnih kola + frekvencija cele istorije do tog trenutka), ansambl proseka tri modela, kalibraciju na treningu i sekundarni kriterijum pri izboru top-7.
Implementirajući bogate obelježja (5 uzastopnih žreba + frekvencija + razmak), isti X u eksperimentu za Loto 7/39, ansambl proseka tri modela za predlog i sekundarni kriterijum preko globalne učestalosti. 
Kalibracija CalibratedClassifierCV sa OneVsRest na 39 izlaza je nezgodna i skupa — preskače se uz kratak komentar u kodu.





5 uzastopnih žreba kao one-hot (V[i] … V[i-4]),
frekvenciju svakog broja 1-39 na svim dosadašnjim žrebovima do tog koraka (V[0]..V[i]),
razmak (koliko je žrebova prošlo od poslednjeg izlaska broja), računato na istoj istoriji.
Za predlog sledećeg koristi se ista logika na celom V (poslednjih pet žreba + freq/gap od svih redova u CSV-u).

Eksperiment (run_experiment)
Isti prošireni X za Loto 7/39; Viking/Euro i dalje samo stari prev_vec.

Predlog kombinacije
Ansambl: prosek predict_proba od sva tri modela (SGD, šuma, XGB/GB).
Istak: ako su verovatnoće blizu, blago se uvlači globalna učestalost u CSV-u (combined).
Kalibracija
CalibratedClassifierCV na OneVsRest x 39 izlaza na ~4500 redova je vrlo težak i često nezgodan; nije uključen — u kodu je kratka napomena u docstringu predikcije.

Tekst ispod predloga na stranici je ažuriran da odgovara novom ponašanju.





Izlaz iz Flask server (python3 run.py) u terminalu:

pritisni Ctrl + C (na Macu isto) — zaustaviće se proces.

vim ili merge poruku: 
Esc, pa :q! (bez čuvanja) 
ili :wq (sa čuvanjem), pa Enter.
"""
