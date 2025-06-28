sumo_env.py – Agentul primește ca observație numărul de vehicule oprite pe fiecare drum și decide, pentru fiecare intersecție semaforizată, care fază să fie activă. 
Am incercat si ca spatiul de observare sa fie doar drumurile ce conduc spre interectii semaforizate, dar a parut ca agentul are learning-rate mai mare daca are acces la toate drumurile.
De asemenea, in environment este definita o metoda custom de facut culoare galben la semafoare, pentru ca altfel agentul schimba foarte des phase-urile si apareau multe evenimente de sudden break si sudden stop.

Ca si completare pentru ce are voie agentul sa faca: exista niste phase-uri scrise de mana in fisierul de netedit al orasului, iar agentul are voie sa aleaga care este phase-ul urmator in fiecare interectie.

Recompensa este calculată in acest moment pe baza mai multor factori: viteza medie a vehiculelor, vehicule ajunse la destinație , număr de opriri, frânări bruște, teleporturi(care se intampla cand o masina este hard-stuck pentru o anumita perioada de timp).

Am incercat si un reward bazat pe intarzieri, care calculeaza timpul total intarziat inainte de o schimbare, si ofera feedback pozitiv daca acesta este minimizat dupa o actiune.(bazat dupa ce am inteles din paper-ul asta: https://www.researchgate.net/publication/324735733_Evaluating_reinforcement_learning_state_representations_for_adaptive_traffic_signal_control)
(nu pare o optiune buna pentru realism, pentru ca nu exista penalizari pentru teleportare, iar agentul poate abuza de acest lucru)

ppo_train.py – inițializează agentul PPO, creează mediul și rulează antrenarea pe un număr mare de timesteps. La final, modelul este salvat, iar logurile sunt disponibile în TensorBoard pentru analiză.

Din ce am citit, ppo este cel mai capabil sa gestioneze mai multe date in acelasi timp. Pe langa asta, am citit recent despre o alta abordare asupra proiectului, ceea ce m-a determinat si sa va intreb legat de functia de reward.
 In momentul de fata, exista un singur agent care controleaza toata intersectiile, iar ideea din capul meu a fost ca in acest mod, este capabil sa faca inlantuiri de verde la semafoare, insa din tot training-ul facut nu pare ca asta sa fie abordarea lui.
 Ce am inteles e ca pentru un astfel de comportament ar fi indicat sa folosesc mai multi agenti care sa comunice intre ei, fiecare agent fiind repartizat pe o singura interectie.

Din rezultatele din TensorBoard, agentul pare ca duce reward-ul la o valoare negativa foarte mica, in prima parte a episoadelor, dupa care incepe sa invete treptat, dar nu pare un rezultat bun pentru ca learning rate-ul este tot mic si cu TIMESTEPS = 2_000_000.
Valorile finale la care ajunge sunt mai slabe decat cele obtinute la simularile facute fara agent adaugat(cele cu training.py).

simulation_generator.py – generează rute aleatorii pentru vehicule și este apelat automat înainte de fiecare episod.
La fiecare episod nou, se genereaza random mai multe caracteristi pentru masini, spre exemplu bias-ul de centru/periferie(pentru a simula rush-hour), numarul total de masini, etc.

In repo este adaugat doar orasul mic, care este mai mult o comuna preluata din Romania ce se comporta ca un oras. Sunt facute in sumo si netedit scenarii pentru un oras mediu si pentru o parte din zona centrala o Bucurestiului(se incadreaza simularea undeva intre Victoriei, Muncii, Universitate, Eroilor)
Acestea 2 sunt mult mai complexe, cu treceri de pietoni, transport in comun, sensuri giratorii, multiple benzi de circulatie.

Agentul a fost testat o data si pe orasul mediu, in ideea ca poate algoritmul functioneaza mai bine pe scenarii complexe, dar tot nu s-a descurcat.

Inca sunt niste junk-files in folder, am incercat sa fac antrenarea mai rapida prin paralelizare, mai exact sa fac cate un episod sa ruleze pe fiecare core de la procesor, dar nu a functionat.
