"""French teaching notes and smaller cells, without changing executable logic."""
from pathlib import Path
import ast
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]

def section(title,purpose,method,result):
    return nb.v4.new_markdown_cell(f'## {title}\n\n**But :** {purpose}\n\n**Méthode et justification :** {method}\n\n**Résultat attendu :** {result}')

def chunks(source,markers):
    starts=[0]+[source.index(m) for m in markers]+[len(source)]
    return [source[a:b].strip() for a,b in zip(starts,starts[1:])]

def enrich(book,name):
    if book.metadata.get('olivesmart_explanations'):return book
    originals=[c.source for c in book.cells if c.cell_type=='code']
    cells=[book.cells[0]]
    def add(title,but,method,result,source):
        cells.extend([section(title,but,method,result),nb.v4.new_code_cell(source)])
    if name.startswith('01'):
        parts=chunks(originals[0],["status=json.loads", "data=pd.read_csv"])
        notes=[
            ('1. Préparer les chemins et les bibliothèques','Définir les entrées du nouveau dataset et le dossier de sortie.','Le notebook part de la racine du projet et ne lance aucun téléchargement. Les anciennes données ne sont pas utilisées.','Les chemins ROOT, DATA et PREP sont disponibles.'),
            ('2. Vérifier la fin de l’import et l’intégrité','S’assurer que le fichier analysé est celui qui a été validé.','Le statut doit indiquer un import terminé sans erreur active. L’empreinte SHA-256 du CSV est comparée au manifeste : une différence demande de refaire la vérification de l’import.','Les assertions passent ; sinon, arrêter la préparation et rechercher la cause.'),
            ('3. Charger, contrôler et décrire les observations','Lire les mesures et distinguer données valides et mois manquants.','Les dates ISO acceptent des fractions de seconde facultatives. Les couples parcelle–mois doivent être uniques. Une observation rejetée ne doit pas avoir de réflectance ou indice exploitable ; le NDVI accepté doit rester entre −1 et 1. Le tableau compte les mesures valides par année. Le nettoyage radiométrique et nuageux a déjà été effectué à l’import : SCL 4/5, au moins 60 % de pixels valides et neuf pixels par polygone. Aucune interpolation n’est ajoutée ici.','Une grille de 60 mois avec des trous explicites, particulièrement nombreux en 2022.')]
        for (title,but,method,result),part in zip(notes,parts):add(title,but,method,result,part)
        parts=chunks(originals[1],["data['NDVI']",'records=[]', 'arrays={}', 'task=PREP/'])
        notes=[
            ('4. Réserver les groupes spatiaux','Tester la généralisation sur des parcelles absentes de l’apprentissage.','Les polygones fortement superposés partagent un groupe. Environ 20 % des groupes de chaque région sont réservés avec un classement SHA-256 reproductible. Aucun de leurs exemples ne passe dans le train ou la validation. Un groupe spatial ne correspond pas forcément à une exploitation indépendante.','Un ensemble holdout de groupes destiné au test spatial final.'),
            ('5. Définir les caractéristiques et la chronologie','Préparer les variables d’entrée et les frontières train/validation/test.','Entrées : NDVI, SAVI, rouge, proche infrarouge et mois encodé en sinus/cosinus. Ce codage rapproche décembre et janvier. Les identifiants ne sont pas des prédicteurs. Les 60 mois sont divisés en 42/9/9 : train janvier 2021–juin 2024 ; validation juillet 2024–mars 2025 ; test avril–décembre 2025. Les indices/réflectances sont partiellement redondants : leur présence ne constitue pas une preuve d’apport indépendant.','La liste des caractéristiques et les frontières temporelles sont fixées.'),
            ('6. Construire les séquences de prévision','Apprendre à prévoir le NDVI du mois suivant à partir de six mois passés.','Chaque exemple contient six observations mensuelles en entrée et un NDVI cible au septième mois. Les sept mois doivent être valides et rester dans le même bloc ; une fenêtre incomplète est exclue, sans remplissage. Le test spatial utilise uniquement les dates du bloc test. Les dates exactes sont conservées : le mois suivant n’implique pas un horizon fixe de 30 jours.','Quatre collections : train, validation, test temporel et test spatial indépendant du train.'),
            ('7. Contrôler les fuites et former les tableaux','Garantir que l’apprentissage n’accède pas au futur ou aux groupes réservés.','Les tableaux X ont la forme (nombre d’exemples, 6 mois, nombre de caractéristiques). Les y sont les NDVI cibles. Les assertions vérifient les effectifs, la chronologie et la séparation des groupes ; la dernière acquisition d’entrée doit précéder l’acquisition cible. Les fenêtres glissantes restent corrélées.','Des tableaux numériques et des métadonnées cohérents ; les ratios finaux d’exemples peuvent différer de 70/15/15 après filtrage.'),
            ('8. Exporter le contrat vers le notebook 02','Rendre la préparation réutilisable dans un kernel neuf.','samples.npz contient X/y bruts ; samples.csv contient unités, groupes, splits et dates. Les manifestes décrivent les variables, limites, effectifs et empreintes des fichiers. La standardisation sera ajustée dans 02 sur le train uniquement, jamais sur la validation ou les tests.','Les fichiers sont écrits dans data/processed/model_inputs_sentinel/. Le tableau final donne les effectifs réellement retenus.')]
        for (title,but,method,result),part in zip(notes,parts):add(title,but,method,result,part)
        add('9. Visualiser la couverture','Repérer les périodes à faible disponibilité.','La fraction de parcelles acceptées est tracée chaque mois. Une chute indique une perte de données exploitables, pas nécessairement une sécheresse. Ce graphique décrit la couverture, pas la santé des oliviers.','Une courbe montrant notamment les lacunes de 2022.',originals[2])
        cells.append(nb.v4.new_markdown_cell('### Passage au notebook 02\n\nUne fois toutes les cellules exécutées sans erreur, ouvrir 02 dans un kernel neuf. Si les sources ou paramètres de préparation changent, réexécuter 01 avant toute nouvelle expérience.'))
    else:
        add('1. Choisir le mode d’exécution','Séparer le contrôle de préparation du lancement coûteux des modèles.','RUN_TRAINING=False charge et vérifie le pipeline. Mettre True pour entraîner. RUN_NAME horodaté distingue les expériences ; conserver le dossier et son protocole pour reproduire les résultats.','Une configuration explicite ; aucun entraînement complet par défaut.',originals[0])
        engine=originals[1]; lines=engine.splitlines(keepends=True)
        nodes=[n for n in ast.parse(engine).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
        start=nodes[0].lineno-1
        add('2. Charger les outils scientifiques','Rassembler les bibliothèques et fixer la graine aléatoire.','scikit-learn fournit les métriques, la normalisation et Random Forest ; XGBoost/LightGBM les modèles boostés ; PyTorch les réseaux. La graine est fixée pour rendre les essais comparables, mais une seule graine ne démontre pas la stabilité statistique.','Les dépendances sont importées. Une erreur d’import doit être résolue avant l’entraînement.',''.join(lines[:start]).strip())
        notes={
            'SequenceNet':('3. Définir LSTM, GRU et CNN-LSTM','Comparer trois architectures temporelles sur les mêmes séquences.','LSTM et GRU résument les six mois par un état caché. CNN-LSTM applique d’abord une convolution temporelle puis une LSTM. Le Dropout limite la dépendance excessive aux unités du réseau ; la sortie est un NDVI continu.','Trois architectures disponibles ; cette cellule définit les réseaux sans les entraîner.'),
            'metrics':('4. Définir les métriques','Quantifier les erreurs de prévision de façon comparable.','MAE : erreur absolue moyenne ; RMSE : pénalise davantage les grosses erreurs ; R² : explique le gain par rapport à une prédiction moyenne sur le jeu évalué et peut être négatif. MAE/RMSE sont exprimées en unités NDVI après inversion de la standardisation. Le code générique prévoit la classification, mais elle n’est pas activée sur cette tâche sans labels.','Des métriques de régression calculées sur chaque split.'),
            'objective':('5. Fixer le critère de sélection','Choisir les paramètres sans regarder les tests.','Pour ce problème de régression, on minimise la RMSE de validation. Les tests temporel et spatial sont réservés au bilan final.','Un critère unique pour comparer les essais.'),
            'plot_predictions':('6. Préparer les graphiques observé/prédit','Comprendre les erreurs au-delà des scores moyens.','Le nuage observé/prédit doit se rapprocher de la diagonale. La seconde vue montre un segment d’exemples ordonnés par unité puis date ; ce n’est pas une courbe temporelle agrégée d’un seul champ. Les fonctions de confusion/ROC existent pour d’autres tâches, mais ne sont pas utilisées ici.','Des figures sauvegardées pour les tests temporel et spatial.'),
            'train_dl':('7. Entraîner avec Early Stopping','Limiter le surapprentissage et comparer des budgets d’entraînement.','À chaque époque, le réseau apprend sur train par mini-lots, puis les pertes train/validation sont mesurées en mode évaluation. AdamW, Dropout, pénalisation des poids et écrêtage du gradient stabilisent l’apprentissage. La meilleure version selon la perte de validation est restaurée ; arrêt après 12 époques sans amélioration suffisante (min_delta=0,0001). Les budgets 50/100/150 sont des plafonds : un arrêt anticipé peut produire moins d’époques.','Un modèle au meilleur état de validation, son historique, l’époque retenue et son temps d’entraînement.'),
            'table':('8. Préparer les tableaux du rapport','Présenter les résultats sous une forme lisible.','Les tableaux HTML regroupent les métriques et les temps ; leur formatage ne modifie pas les calculs.','Des tableaux utilisables dans le rapport final.'),
            'main':('9. Orchestrer la comparaison complète','Appliquer le même protocole à la baseline et aux six familles de modèles.','Le détail de l’enchaînement est expliqué ci-dessous. Cette cellule définit main() ; le lancement effectif se fait à l’étape suivante.','Un pipeline prêt, dont les sorties restent séparées par expérience.')}
        for i,node in enumerate(nodes):
            if node.name=='main':
                cells.append(nb.v4.new_markdown_cell('''### Déroulement de la comparaison dans `main()`

1. **Intégrité et chargement** : vérifier les empreintes, charger les quatre splits préparés dans 01 et enregistrer les versions des bibliothèques.
2. **Normalisation** : ajuster StandardScaler des caractéristiques et de la cible uniquement sur train. Réutiliser ces paramètres pour les autres splits et sauvegarder les transformations. Les modèles ML reçoivent les mêmes six mois aplatis ; les modèles DL conservent la dimension temporelle.
3. **Baseline de persistance** : prévoir que le prochain NDVI sera égal au dernier NDVI observé. Un modèle complexe doit montrer un gain utile face à cette référence.
4. **Recherche ML** : essayer trois configurations de profondeur/taux d’apprentissage par famille (150 arbres), puis retenir la plus faible RMSE de validation. La forêt n’utilise pas de taux d’apprentissage.
5. **Recherche DL** : pour chaque architecture et plafond de 50/100/150 époques, essayer deux configurations (16 unités, lr=0,001 ; 32 unités, lr=0,003), avec Early Stopping. Les paramètres retenus et les époques réellement exécutées sont enregistrés.
6. **Évaluation** : calculer MAE, RMSE et R² sur train, validation, test temporel et groupes spatiaux réservés. Les tests ne sélectionnent ni paramètres ni budgets. Tous les modèles utilisent les mêmes exemples.
7. **Coûts** : mesurer le temps de l’essai retenu, le temps total de recherche et la médiane de cinq inférences sur le lot test, rapportée par exemple. Ces temps CPU dépendent du matériel et excluent le prétraitement de l’inférence.
8. **Surapprentissage** : comparer les scores train/validation et les courbes de loss. Un écart peut aussi venir du changement de saison/distribution ; ce n’est pas une preuve unique de surapprentissage.
9. **Choix final et sauvegarde** : choisir le modèle global par RMSE de validation, baseline comprise, puis présenter ses tests. Enregistrer modèles, transformations, prédictions, essais, courbes et protocole dans `outputs/benchmarks/<RUN_NAME>/`.

La grille est limitée et une seule graine est utilisée : la comparaison est expérimentale. Elle ne constitue ni une preuve de supériorité statistique ni un diagnostic validé de stress hydrique.
'''))
            end=nodes[i+1].lineno-1 if i+1<len(nodes) else len(lines)
            add(*notes[node.name],''.join(lines[node.lineno-1:end]).strip())
        add('10. Vérifier les entrées et lancer si demandé','Éviter de lancer des modèles sur des données périmées ou modifiées.','Les fichiers exportés par 01 doivent exister et leurs empreintes doivent correspondre au manifeste. RUN_TRAINING=True appelle main() ; False affiche les effectifs et confirme que le pipeline est prêt. Une préparation modifiée implique une nouvelle expérience.','Confirmation des entrées ; puis, seulement si activé, entraînement de la comparaison.',originals[2])
        add('11. Lire les résultats et les courbes','Choisir un modèle utile pour OliveSmart et identifier ses limites.','Comparer séparément test temporel et test spatial. Chercher des erreurs faibles, un gain face à la baseline et des coûts raisonnables. R² négatif ou absence de gain n’est pas un échec du pipeline : le modèle complexe n’est alors pas justifié. Les courbes affichent les époques réellement exécutées. Le rapport HTML contient les figures observé/prédit ; il est aussi consultable depuis le dossier de sortie.','Tableaux de comparaison, rapport et courbes après l’entraînement. En mode False, cette cellule ne produit pas de résultats de modèles.',originals[3])
    # Splitting and comments must not change any executable statements.
    before=ast.dump(ast.parse('\n\n'.join(originals)),include_attributes=False)
    after=ast.dump(ast.parse('\n\n'.join(c.source for c in cells if c.cell_type=='code')),include_attributes=False)
    assert before==after,'Executable logic changed during documentation edit'
    book.cells=cells
    book.metadata['olivesmart_explanations']=True
    return book

if __name__=='__main__':
    for name in ['01_nettoyage_pretraitement.ipynb','02_entrainement_resultats.ipynb']:
        book=enrich(nb.read(ROOT/name,as_version=4),name)
        nb.validate(book);nb.write(book,ROOT/name)
        print(name,len(book.cells),'cells; executable logic unchanged')
