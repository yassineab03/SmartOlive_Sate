"""Rapport d'avancement à partager avec l'équipe, format A4."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'; OUT.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialBold', 'C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialBold',italic='Arial',boldItalic='ArialBold')
olive=colors.HexColor('#405B2E'); ink=colors.HexColor('#233125')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyFR',fontName='Arial',fontSize=10,leading=14,textColor=ink,spaceAfter=7))
styles.add(ParagraphStyle(name='TitleFR',fontName='ArialBold',fontSize=24,leading=28,textColor=olive,spaceAfter=8))
styles.add(ParagraphStyle(name='SubFR',fontName='Arial',fontSize=11,leading=15,textColor=ink,spaceAfter=13))
styles.add(ParagraphStyle(name='HeadFR',fontName='ArialBold',fontSize=12,leading=16,textColor=olive,spaceBefore=10,spaceAfter=6))
styles.add(ParagraphStyle(name='CellFR',fontName='Arial',fontSize=9,leading=12,textColor=ink))
styles.add(ParagraphStyle(name='SmallFR',fontName='Arial',fontSize=8,leading=11,textColor=ink,spaceAfter=5))
story=[]
def p(text,style='BodyFR'): return Paragraph(text,styles[style])
def add(text,style='BodyFR'): story.append(p(text,style))
def table(rows,widths):
    cells=[[p(str(c),'CellFR') for c in row] for row in rows]
    t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6EDDF')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,0),(-1,0),.7,olive),('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#D6DED1'))]))
    story.append(t); story.append(Spacer(1,8))

add('OliveSmart','TitleFR')
add("Rapport d’avancement IA | Analyse satellitaire multi-temporelle<br/>Sujet OLV-4 - Situation au 2 novembre 2026",'SubFR')
add('1. À quoi sert le projet ?','HeadFR')
add("OliveSmart est une plateforme de gestion intelligente des oliveraies marocaines. Cette première phase IA vise à repérer les <b>variations inhabituelles de vigueur de la végétation par parcelle</b>, à partir d’images satellites prises à plusieurs dates, en les distinguant des variations saisonnières normales.")
add("Le résultat attendu est une alerte invitant à vérifier une parcelle. Une alerte ne constitue pas, à elle seule, un diagnostic de maladie ou de manque d’eau. La partie IA pourra ensuite alimenter la plateforme en microservices.")
add('2. Organisation des tâches IA','HeadFR')
add("<b>Collecte</b> des images, des limites parcellaires et de la météo → <b>contrôle qualité</b> et masquage des nuages → <b>calcul des indices</b> NDVI, NDRE, NDWI et SAVI → <b>séries temporelles</b> et saisonnalité → <b>comparaison des modèles</b> → alertes et évaluation.")
table([['Méthode','Famille','Rôle dans la comparaison'],
 ['Z-score NDVI','Statistique','Référence simple : baisse par rapport au cycle attendu.'],
 ['Isolation Forest','Machine learning','Repérage des fenêtres multivariées atypiques.'],
 ['One-Class SVM','Machine learning','Apprentissage d’une frontière de normalité.'],
 ['Autoencoder','Deep learning','Détection par erreur de reconstruction.']],[110,110,291])
add("Protocole préparé : transformations ajustées sur l’apprentissage, optimisation TPE sur validation temporelle, test sur période future et, si disponible, bassin distinct. Indicateurs prévus : précision, rappel, F1, délai, faux positifs par parcelle et stabilité interannuelle.")
add('3. Ce qui est réalisé','HeadFR')
add("Le dossier <b>SmartOlive_Sate</b> contient le notebook Python pour PyCharm, les modèles, l’optimisation et les fonctions d’évaluation. Les tests techniques des quatre modèles ont été réalisés sur de petites données artificielles, sans les présenter comme des performances agronomiques.")
add("Deux ZIP Sentinel-2 L2A ont été téléchargés depuis Copernicus et vérifiés. Une zone de test GeoJSON d’environ <b>117 hectares</b> a été enregistrée. Les quatre indices ont été extraits localement, <b>sans Earth Engine</b>, après correction radiométrique et filtrage qualité.")
table([['Acquisition','Pixels passant le filtre QA','NDVI moyen de la zone'],
 ['17 septembre 2024','99,97 %','0,585'],['1er novembre 2024','100 %','0,725']],[170,170,171])
add("<b>Limite actuelle :</b> deux dates ne suffisent pas pour entraîner les modèles temporels. La présence d’oliviers et les limites des parcelles ne sont pas confirmées. Les valeurs ci-dessus sont descriptives ; aucune anomalie n’a été validée.",'SmallFR')

story.append(PageBreak())
add('Données à compléter','TitleFR')
add("Recherche à répartir entre les membres de l’équipe",'SubFR')
add('4. Jeux de données à rechercher','HeadFR')
table([['Source / état','Ce que l’équipe doit obtenir','Pourquoi ?'],
 ['<b>Sentinel-2 L2A</b><br/>Deux acquisitions traitées',
  'Images des mêmes parcelles sur plusieurs années, bandes numériques, masques SCL et métadonnées. Première exploration : avril à novembre 2024. Pour la zone de test actuelle : tuile T29SPR.',
  'Former des séries régulières, permettre des agrégations de 5 à 15 jours et apprendre plusieurs cycles annuels.'],
 ['<b>Landsat Collection 2 Level-2</b><br/>Pas encore importé',
  'Historique sur la même zone, réflectances de surface, dates, masques QA_PIXEL et QA_RADSAT.',
  'Étudier la saisonnalité à long terme à 30 m. Conserver une baseline distincte de Sentinel avant toute harmonisation.'],
 ['<b>ERA5-Land</b><br/>Pas encore importé',
  'Température et précipitations pour la zone et les périodes étudiées, avec unités et dates.',
  'Ajouter le contexte météo et comparer les modèles avec et sans météo.'],
 ['<b>Collecte Maroc : parcelles</b><br/>Zone exploratoire seulement',
  'Polygones GeoJSON de véritables oliveraies, identifiants stables de parcelle, exploitation et bassin ; confirmation de la culture.',
  'Analyser des parcelles fiables et organiser les séparations temporelles et spatiales.'],
 ['<b>Collecte Maroc : terrain</b><br/>À constituer',
  'Visites datées, stress confirmé, périodes normales vérifiées, irrigation, taille et date de disponibilité des observations ; périodes de surveillance exhaustive.',
  'Vérifier les alertes et calculer précision, rappel, F1, délai et faux positifs.']],[112,238,161])
add('5. Consignes pour la collecte','HeadFR')
add("• Télécharger les <b>produits numériques originaux</b>, pas des captures d’écran ni des images colorisées.<br/>• Couvrir les <b>mêmes parcelles</b> aux différentes dates et conserver les métadonnées.<br/>• Ne pas remplacer un événement inconnu par une étiquette « normal ».<br/>• Distinguer données confirmées, données indirectes et zones exploratoires.")
add("<b>Priorité :</b> identifier des parcelles marocaines réellement plantées d’oliviers et réunir plusieurs années d’images sur ces parcelles. Sans terrain, nous pourrons produire des scores et des alertes, mais pas démontrer leur fiabilité agronomique.")
add('Sources officielles','HeadFR')
add('Sentinel-2 / Copernicus : <link href="https://dataspace.copernicus.eu/" color="#405B2E">dataspace.copernicus.eu</link><br/>Landsat : <link href="https://www.usgs.gov/landsat-missions/landsat-collection-2" color="#405B2E">USGS - Landsat Collection 2</link><br/>ERA5-Land : <link href="https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR" color="#405B2E">Catalogue officiel ERA5-Land Daily Aggregated</link>','SmallFR')
add("Copernicus est l’accès retenu pour Sentinel-2. Earth Engine reste facultatif ; la chaîne locale actuelle traite Sentinel-2 uniquement. Les accès locaux à Landsat et ERA5-Land restent à mettre en place.",'SmallFR')

def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#D6DED1')); c.line(42,36,A4[0]-42,36)
    c.setFont('Arial',8); c.setFillColor(olive)
    c.drawString(42,23,'OliveSmart | Point équipe - 02/11/2026')
    c.drawRightString(A4[0]-42,23,str(doc.page))

target=OUT/'OliveSmart_Rapport_avancement_IA.pdf'
doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=35,bottomMargin=46,
                      title='OliveSmart - Rapport d’avancement IA',author='Équipe OliveSmart')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(target)
assert len(reader.pages)==2, f'Mise en page inattendue : {len(reader.pages)} pages'
assert all(page.extract_text().strip() for page in reader.pages)
print(target)
print('Pages :',len(reader.pages))
