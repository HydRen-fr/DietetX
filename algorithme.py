# Algorithme DietetX

import re
import pandas
from functools import reduce
import operator
import nltk
import spacy
from statistics import mean
import unicodedata
import difflib
from nltk.stem.snowball import SnowballStemmer
from collections import OrderedDict
import copy

from nltk.corpus import *
from spacy import *

from prettytable import PrettyTable

nltk.download("stopwords")

# python -m spacy download fr_core_news_sm --> deja en zip


# FONCTIONS

def ll_str(ll):
    ll = [" ".join(ele) for ele in ll]
    ll = " ".join(ll)
    return ll


def ll_l(ll):
    ll = reduce(operator.concat, ll)
    return ll


def ll_l2(ll):
    ll = [' '.join(ele) for ele in ll]
    return ll


def strip_acc(s_acc):
    s_no_acc = ''.join((c for c in unicodedata.normalize(
        'NFD', s_acc) if unicodedata.category(c) != 'Mn'))
    return s_no_acc


def lt_ll(lt):
    ll = [list(ele) for ele in lt]
    return ll


def ll_str_int(lls):
    lli = [list(map(int, x)) for x in lls]
    return lli


def average_ll(ll):
    result = [mean(x) for x in ll]
    return result


def return_stem(sentence):
    doc = nlp(sentence)
    return [stemmer.stem(X.text) for X in doc]


def strip_end(text, suffix):
    l_txt = text.split()
    for i in range(len(l_txt)):
        if suffix and l_txt[i].endswith(suffix):
            l_txt[i] = l_txt[i][:-len(suffix)]

    text = " ".join(l_txt)
    return text


# NETTOYAGE

nlp = spacy.load("fr_core_news_sm")
stopwords = nltk.corpus.stopwords.words('french')
stopwords = [strip_acc(i) for i in stopwords]
stemmer = SnowballStemmer(language='french')

class MenageStr():
  def __init__(self, test_str):
    self.rmpls = {"(?!(?<=\d).(?=\d))[^\w\s ,'’\/ \- ]":"", # PONCTUATION --> RIEN (exceptions [^...] et on fait attention aux points des décimaux)
            "(?<=\d),(?=\d)":".", # VIRGULES DES CHIFFRES DEVIENNENT DES POINTS POUR PYTHON
            "œ":"oe", # E DANS L'O
            # POUR LES UNITES
            r"\bcuillere(s)?\s*(a)?\s*cafe\b":"qirc",
            r"\bcuillere(s)?\s*(a)?\s*(soupe)?\b":"qirs",
            r"\bpoignee(s)?\b":"qirs",
            r"\bc\s*(a)?\s*s\b":"qirs",
            r"\bc\s*(a)?\s*c\b":"qirc",
            r"\bk(ilo)?(s)?\b":"kg",
            r"\bg(r)?(ramme)?(s)?\b":"g",
            "['’\/ \- ]":" "} # METTRE UN ESPACE POUR EPURER

    self.pas_normaliser = ["maïs", "maïs,"] # si le mot est collé à une virgule --> moyen le plus simple

    self.pres = ("facultatif", "environ", "petit", "grand", "gros", "bouquet", "plat", "gousse", "sel", "poivre", "pave", "papier", "aluminium", "surgel",
            "fraich", "quelqu", "rondell", "filet", "moulu", "peu", "choix", "melange", "taille","moyen", "concas", "ajout", "moelleux", "decortiq", "cm", "bombee", "entier", "nature")
    self.entire_pres = ["frais", "blanche", "cuit", "hachee"]

    # Ce que je veux/peux pas faire pour l'instant:
    # Les ou, virer les surplus de chiffres (piste : (?<=\d)(?:.*)(\d[^,]+))

    self.test_str = test_str

    self.clean_str = self.menage()

  def menage(self):
    # ESPACES DE TROP, PARENTHESES, END DOTS, LOWERCASE, NORMALISATION, STR|INT
      self.test_str = re.sub(' +', ' ', self.test_str)
      self.test_str = re.sub(r'\([^)]*\)', '', self.test_str)
      self.test_str = re.sub(r"\.(?=\s[A-Z 0-9])",  ",", self.test_str)
      self.test_str = self.test_str.lower()

      mots = self.test_str.split()
      mots = [strip_acc(mot) if mot not in self.pas_normaliser else mot for mot in mots]
      self.test_str = " ".join(mots)
      
      self.test_str = re.sub(r"\b(\d+)([a-z]+)\b", r"\1 \2", self.test_str)
      
      
      # REMPLACEMENTS (ET MOY)
      for k, v in self.rmpls.items():
        self.test_str = re.sub(k, v, self.test_str)
      a_nums = average_ll(ll_str_int(lt_ll(re.findall(r"\b(\d+(?:\.\d+)?)(?:[^\w.]+(?:\w+[^\w,]+){0,1}(\d+(?:\.\d+)?))+\b", self.test_str))))
      self.test_str = re.sub(r"\b(\d+(?:\.\d+)?)(?:[^\w.]+(?:\w+[^\w,]+){0,1}(\d+(?:\.\d+)?))+\b",  lambda match: str(a_nums.pop(0)), self.test_str)

      # BLOCS VIRGULES
      begin = re.split(',+', self.test_str)

      # BLOC SPLITES
      for i in range(len(begin)):
        begin[i] = begin[i].split()

      # STOPWORDS
      for i in range(len(begin)):
        begin[i] = [j for j in begin[i] if j not in stopwords or j == "l"]

      # PRECISIONS INUTILES
      for i in range(len(begin)):
        begin[i] = [j for j in begin[i] if j.startswith(self.pres) == False]
      for i in range(len(begin)):
        begin[i] = [j for j in begin[i] if j not in self.entire_pres]

      # DUPLIQUES
      for i in range(len(begin)):
        begin[i] = list(OrderedDict.fromkeys(begin[i]))

      begin = [i for i in begin if i] # PAS D'ITEMS VIDES
      return begin


class MenageListFood():
  def __init__(self, list_food):
    # Même idée MAIS on part directement de la liste bien tokenizée et on n'enlève les précisions inutiles
    self.rmpls2 = {"(?!\b[^\w\s]\b)[^\w\s '’ - ]":"", 
              "œ":"oe", # E DANS L'O
              "[%'’ - ]":" "}

    self.pas_normaliser2 = ["maïs"]

    self.pres2 = ("appertis", "preemball", "rechauff", "reconstit", "cuit", "fait", "maison", "surgel", "fraich", "denoyaut", "egoutt", "nature", "enrichi")
    self.entire_pres2 = ["frais", "pulpe", "cru", "crue", "peau", "pepin", "pepins", "sale", "salee", "mg", "matiere", "grasse", "rayon", "precision"]
    self.list_food = list_food

    self.clean_list_food = self.menage2()

  def menage2(self):
    # ESPACES DE TROP, PARENTHESES, END DOTS, LOWERCASE, NORMALISATION, STR|INT, REMPLACEMENTS
    for i in range(len(self.list_food)):
      self.list_food[i] = re.sub(' +', ' ', self.list_food[i])
      self.list_food[i] = re.sub(r'\([^)]*\)', '', self.list_food[i])
      self.list_food[i] = self.list_food[i].lower()

      mots = self.list_food[i].split()
      mots = [strip_acc(mot) if mot not in self.pas_normaliser2 else mot for mot in mots]
      self.list_food[i] = " ".join(mots)

      self.list_food[i] = re.sub(r"\b(\d+)([a-z]+)\b", r"\1 \2", self.list_food[i])
      for k, v in self.rmpls2.items():
        self.list_food[i] = re.sub(k, v, self.list_food[i])

    # BLOC SPLITES
    for i in range(len(self.list_food)):
      self.list_food[i] = self.list_food[i].split()

    # STOPWORDS
    for i in range(len(self.list_food)):
      self.list_food[i] = [j for j in self.list_food[i] if j not in stopwords]

    # PRECISIONS INUTILES
    for i in range(len(self.list_food)):
      self.list_food[i] = [j for j in self.list_food[i] if j.startswith(self.pres2) == False]
    for i in range(len(self.list_food)):
      self.list_food[i] = [j for j in self.list_food[i] if j not in self.entire_pres2]

    self.list_food = [i for i in self.list_food if i] # PAS D'ITEMS VIDES
    self.list_food = ll_l2(self.list_food)
    return self.list_food


class MenageListNutri():
  def __init__(self, list_nutri):
    # On prend une liste de chiffres français et on convertit en format python
    self.rmpls3 = {"(?<=\d),(?=\d)":".", "[^\d \.]":""}

    self.list_nutri = list_nutri

    self.clean_list_nutri = self.menage3()

  def menage3(self):
    # ESPACES DE TROP, PARENTHESES, NON-STR, REMPLACEMENTS
    for i in range(len(self.list_nutri)):
      self.list_nutri[i] = str(self.list_nutri[i])
      self.list_nutri[i] = re.sub(' +', ' ', self.list_nutri[i])
      self.list_nutri[i] = re.sub(r'\([^)]*\)', '', self.list_nutri[i])

      for k, v in self.rmpls3.items():
        self.list_nutri[i] = re.sub(k, v, self.list_nutri[i])

    return self.list_nutri


class Isolation():
  def __init__(self, rd):
    self.units = ["g", "kg", "ml", "cl", "l", "qirc", "qirs", "louche", "tasse", "bol", "verre"]
    self.rd = rd
    self.rd_d = self.isolation()

  def isolation(self):
    # RD_D
    self.rd_d = []
    for i in range(len(self.rd)):
      self.rd_d.append({"Quantity":[], "Unity":[], "Product":[]})

    for i in range(len(self.rd)):

    # QUANTITES
      for j in range(len(self.rd[i])):
        try:
          if float(self.rd[i][j]):
            self.rd_d[i]["Quantity"].append(self.rd[i][j])
        except ValueError:
          pass
      self.rd[i] = [j for j in self.rd[i] if j not in self.rd_d[i]["Quantity"]]

    # UNITES
      self.rd_d[i]["Unity"] = [j for j in self.rd[i] if j in self.units]
      self.rd[i] = [j for j in self.rd[i] if j not in self.rd_d[i]["Unity"]]

    # LE RESTE EN PRODUITS
      self.rd_d[i]["Product"] = [j for j in self.rd[i]]
      self.rd_d[i]["Product"] = " ".join(self.rd_d[i]["Product"]) # LISTE --> STR
      self.rd[i] = [j for j in self.rd[i] if j not in self.rd_d[i]["Product"]]

    self.rd_d = [i for i in self.rd_d if i["Product"] and i["Quantity"]] # PRODUITS DONT ON CONNAIT LES QUANTITES

    for i in range(len(self.rd_d)):
      self.rd_d[i]["Quantity"][0] = float(self.rd_d[i]["Quantity"][0]) # FLOATS

    return self.rd_d
  
class Conversion():
    def __init__(self, rd_d, dict_poids, pairs_infos):
        self.rd_d = copy.deepcopy(rd_d)
        self.spe_units_conversion = {"qirc":5, "qirs":15, "louche":150, "verre":180, "tasse":130, "bol":280}
        self.other_units_conversion = {"ml":1,"cl":10, "l":1000, "kg":1000}
        self.pairs_infos = pairs_infos

        self.dict_poids = dict_poids

        self.no_unity_fix()
        self.spe_unity_fix()
        self.other_unity_fix()

    def get_pair(self, prod, l_comp):

      # VARIABLES 
      coff_val = 1.0 
      pairs = [] 
      b_pair = "" 
      found = False
      stemmed_prod = return_stem(prod)

      # PROCESS
      while not pairs and coff_val >= 0 and found == False:

        pairs = difflib.get_close_matches(prod, l_comp, cutoff=coff_val)

        # SI ON A QQCH ON CHERCHE LE BON ITEM
        if pairs:

          # STEMMING + COMPARAISON
          pairs_set_score = []
          for i in range(len(pairs)):
            stemmed_item = return_stem(pairs[i])

            # LISTE DE SCORES DE RESSEMBLANCE
            pairs_set_score.append(len(set(stemmed_prod).intersection(stemmed_item)))

          # MEILLEURS ITEMS > 0
          if max(pairs_set_score) > 0:
            b_pair = pairs[pairs_set_score.index(max(pairs_set_score))]
            found = True
            break # On a mis found à True mais la boucle while ne se stoppe pas automatiquement, il faut break en plus

          # MOINS EXIGENT SI TOUT EST A 0
          else:
            coff_val -= 0.1
            pairs = []

        # MOINS EXIGENT SI AUCUN MATCH
        else:
          coff_val -= 0.1

      return b_pair

    def no_unity_fix(self): # noms de fonctions explicites
        for i in range(len(self.rd_d)):
            if not self.rd_d[i]["Unity"]:
                self.rd_d[i]["Unity"].append("g")
                coeff_q = self.rd_d[i]["Quantity"][0]
                self.rd_d[i]["Quantity"] = []
                try:
                    self.rd_d[i]["Quantity"].append(self.dict_poids[self.get_pair(self.rd_d[i]["Product"], self.dict_poids.keys())] * coeff_q)
                except:
                    self.pairs_infos[i]["Index"][0] = "N"
        return self.rd_d

    def spe_unity_fix(self):
        for i in range(len(self.rd_d)):
            if self.rd_d[i]["Unity"][0] in self.spe_units_conversion:
                spe_u_i = self.rd_d[i]["Unity"][0]
                self.rd_d[i]["Unity"] = []
                self.rd_d[i]["Unity"].append("g")
                coeff_q = self.rd_d[i]["Quantity"][0]
                self.rd_d[i]["Quantity"] = []
                self.rd_d[i]["Quantity"].append(self.spe_units_conversion[spe_u_i] * coeff_q)
        return self.rd_d

    def other_unity_fix(self):
        for i in range(len(self.rd_d)):
            if self.rd_d[i]["Unity"][0] in self.other_units_conversion:
                other_u_i = self.rd_d[i]["Unity"][0]
                self.rd_d[i]["Unity"] = []
                self.rd_d[i]["Unity"].append("g")
                coeff_q = self.rd_d[i]["Quantity"][0]
                self.rd_d[i]["Quantity"] = []
                self.rd_d[i]["Quantity"].append(self.other_units_conversion[other_u_i] * coeff_q)
        return self.rd_d

    def convert(self):
        self.rd_d = self.no_unity_fix()
        self.rd_d = self.spe_unity_fix()
        self.rd_d = self.other_unity_fix()
        return self.rd_d
    

class Calcul():
    def __init__(self, rd_d_f, nutriments, pairs_infos):
        self.rd_d_f = rd_d_f
        self.nutriments = nutriments
        self.pairs_infos = pairs_infos
        self.recette_table = self.table_blanche()
        self.all_rows = []
        for i in range(len(rd_d_f)):
            self.all_rows.append([])

        self.recette_table = self.afficher_table()

    def table_blanche(self):
        colonnes = ["Aliments"]
        for k, v in self.nutriments.items():
            colonnes.append(k)
        return PrettyTable(colonnes)

    def get_rows_ll(self):
        for i in range(len(self.rd_d_f)):
            self.all_rows[i].append(self.rd_d_f[i]["Product"])
              
            index_nutrition = self.pairs_infos[i]["Index"][0]

            if index_nutrition != "N":
                coeff_100_g = self.rd_d_f[i]["Quantity"][0] / 100
            
                for k, v in self.nutriments.items():
                    try:
                        self.all_rows[i].append(round(float(v[index_nutrition]) * coeff_100_g, 2))
                    except ValueError:
                        self.all_rows[i].append("N")
            else:
                for k, v in self.nutriments.items():
                    self.all_rows[i].append("N")

    def get_total_row(self):
        total = []
        total.append("TOTAL")

        list_zipped_all_rows = list(zip(*self.all_rows))
        for ele in list_zipped_all_rows[1:]:
            sum_ele = 0
            for item in ele:
                try:
                    sum_ele += item
                except:
                    pass
            total.append(round(sum_ele, 2))

        self.all_rows.append(total)

    def add_rows(self):
        for i in range(len(self.all_rows)):
            self.recette_table.add_row(self.all_rows[i])

    def afficher_table(self):
        self.get_rows_ll()
        self.get_total_row()
        self.add_rows()
        
        return self.recette_table


food_df = pandas.read_excel('static/files/Table_Ciqual_2020.xls')
Products_info = food_df.to_dict('records') # Liste de dictionnaires

# Aliments aux noms trop longs et mal reconnaissables
aliments_complexes = {"Lait demi-écrémé, UHT":"lait", "Huile d'olive vierge extra":"Huile d'olive",
                      "Champignon de Paris ou champignon de couche, bouilli/cuit à l'eau":"Champignon de Paris",
                      "Coeur, poulet, cru":"Coeur volaille", "Amidon de maïs ou fécule de maïs":"Maizena",
                      "Crème de lait, 30% MG, épaisse, rayon frais":"Crème épaisse", "Crème épaisse":"Crème", "Bar commun ou loup, cru, sans précision":"Bar",
                      "Fromage blanc nature ou aux fruits (aliment moyen)":"Fromage blanc", "Oeuf, cru":"Oeufs", "Céleri branche, cru":"Branches céleri",
                      "Thym, séché":"Branches thym", "Farine de blé tendre ou froment T65":"Farine", "Farine":"Farine blé", "Tomate, concentré, appertisé":"Concentré tomates",
                      "Sucre blanc":"Sucre", "Sucre":"Sucre en poudre", "Melon cantaloup (par ex.: Charentais, de Cavaillon) pulpe, cru":"Melon", "Nectarine ou brugnon, pulpe et peau, crue":"Nectarine",
                      "Clémentine ou Mandarine, pulpe, crue":"Clémentine", "Clémentine":"Mandarine", "Pomelo (dit Pamplemousse), pulpe, cru":"Pamplemousse", 
                      "Fruit de la passion ou maracudja, pulpe et pépins, cru":"Fruit de la passion", "Fruit de la passion":"Maracudja", "Pruneau, sec":"Pruneaux",
                      "Chou-fleur, cuit":"Chou fleur", "Radis rouge, cru":"Radis", "Potiron, appertisé, égoutté":"Potirons", "Semoule de blé dur, cuite, non salée":"Semoule",
                      "Orge perlée, bouilli/cuite à l'eau, non salée":"Orge", "Boisson au soja, nature, non enrichie, préemballée":"Lait soja", "Lait de coco ou Crème de coco":"Lait coco",
                      "Boisson à l'amande, sucrée, enrichie en calcium, préemballée":"Lait amande", "Boisson à base d'avoine, nature, préemballée":"Lait avoine",
                      "Poulet (var. blanc), viande et peau, cru":"Poulet", "Poulet":"Blanc poulet", "Clam, Praire ou Palourde, bouilli/cuit à l'eau":"Palourde", "Crevette royale rose, cuite":"Gambas",
                      "Sucre roux":"Sucre brun", "Sucre brun":"Cassonade", "Pâte à tartiner chocolat et noisette":"Pâte tartiner", "Beurre à 82% MG, doux":"Beurre",
                      "Matière grasse ou graisse végétale solide (type margarine) pour friture":"Margarine"}
# Aliments préférables de supprimer  
aliments_suppr = ["Groseille à maquereau, crue"]

for i, d in enumerate(Products_info):

  for alisuppr in aliments_suppr:
    if d["alim_nom_fr"] == alisuppr:
      Products_info.remove(d)

  for ac, bon_nom in aliments_complexes.items():
    if d["alim_nom_fr"] == ac:
        nouveau_dict_bon_nom = d.copy()
        nouveau_dict_bon_nom["alim_nom_fr"] = bon_nom
        Products_info.insert(i+1, nouveau_dict_bon_nom)

food = []
for i in range(len(Products_info)):
    food.append(Products_info[i]['alim_nom_fr'])
food = MenageListFood(food)
newFood = food.clean_list_food # NOMS D'ALIMENTS STOCKES ET NETTOYES


calories_kcal = []
for i in range(len(Products_info)):
    calories_kcal.append(Products_info[i]['Energie, Règlement UE N° 1169/2011 (kcal/100 g)'])

acides_gras_satures_g = []
for i in range(len(Products_info)):
    acides_gras_satures_g.append(Products_info[i]['AG saturés (g/100 g)'])

glucides_g = []
for i in range(len(Products_info)):
    glucides_g.append(Products_info[i]['Glucides (g/100 g)'])

proteines_g = []
for i in range(len(Products_info)):
    proteines_g.append(Products_info[i]['Protéines, N x 6.25 (g/100 g)'])

lipides_g = []
for i in range(len(Products_info)):
    lipides_g.append(Products_info[i]['Lipides (g/100 g)'])

eau_g = []
for i in range(len(Products_info)):
    eau_g.append(Products_info[i]['Eau (g/100 g)'])

sucre_g = []
for i in range(len(Products_info)):
    sucre_g.append(Products_info[i]['Sucres (g/100 g)'])

sel_g = []
for i in range(len(Products_info)):
    sel_g.append(Products_info[i]['Sel chlorure de sodium (g/100 g)'])

fibres_g = []
for i in range(len(Products_info)):
    fibres_g.append(Products_info[i]['Fibres alimentaires (g/100 g)'])

cholesterol_mg = []
for i in range(len(Products_info)):
    cholesterol_mg.append(Products_info[i]['Cholestérol (mg/100 g)'])

sodium_mg = []
for i in range(len(Products_info)):
    sodium_mg.append(Products_info[i]['Sodium (mg/100 g)'])


nutriments = {"Calories (kcal)":calories_kcal, "Acides gras saturés (g)":acides_gras_satures_g, "Glucides (g)":glucides_g,
              "Protéines (g)":proteines_g, "Lipides (g)":lipides_g, "Eau (g)":eau_g, "Sucre (g)":sucre_g, "Sel (g)":sel_g, "Fibres (g)":fibres_g,
              "Cholesterol (mg)":cholesterol_mg, "Sodium (mg)":sodium_mg}

for k, v in nutriments.items(): # SOTCKES ET NETTOYES DANS DES LISTES
  v_pas_clean = MenageListNutri(v)
  v = v_pas_clean.clean_list_nutri
        
poids_df = pandas.read_excel('static/files/poids_moyen_g.ods')
dict_poids = poids_df.set_index('nom_aliment').to_dict()['poids_moyen']


''' Tests

Fruits : 1 banane, 2 pommes, 3 poires, 4 pêches, 5 nectarines, 6 abricots, 7 mangues, 8 papayes, 9 goyaves, 10 ananas, 11 kiwis, 12 fraises, 13 framboises, 
14 myrtilles, 15 mûres, 16 cerises, 17 raisins, 18 grenades, 19 oranges, 20 citrons, 21 citrons verts, 22 clémentines, 23 mandarines, 24 kumquats, 25 litchis, 26 pamplemousses, 
27 melons, 28 pastèques, 29 groseilles, 30 groseilles à maquereau, 31 cassis, 32 fruits de la passion, 33 prunes, 34 pruneaux, 35 raisins secs, 36 figues, 37 dattes, 38 abricots secs, 
39 mangues séchées, 40 papayes séchées, 41 ananas séchés, 42 kiwis séchés, 43 pommes séchées, 44 poires séchées, 
45 noix de coco râpée, 46 noix de cajou, 47 noisettes, 48 amandes, 49 pistaches, 50 graines de grenade.

Légumes : 1 carotte, 2 pommes de terre, 3 courgettes, 4 aubergines, 5 poivrons rouges, 6 poivrons jaunes, 7 poivrons verts, 8 tomates, 
9 concombres, 10 oignons rouges, 11 oignons jaunes, 12 échalotes, 13 ail, 14 chou-fleur, 15 brocoli, 16 choux de Bruxelles, 17 chou rouge, 18 chou vert, 19 céleri-rave, 
20 céleri-branche, 21 radis, 22 navets, 23 panais, 24 topinambours, 25 patates douces, 26 courges, 27 potirons, 28 potimarrons, 29 champignons de Paris, 30 girolles, 31 cèpes, 
32 pleurotes, 33 shiitakes, 34 haricots verts, 35 haricots blancs, 36 lentilles, 37 pois chiches, 
38 pois cassés, 39 fèves, 40 artichauts, 41 asperges, 42 fenouil, 43 betteraves, 44 rutabagas, 45 bettes, 46 épinards, 47 salade, 48 endives, 49 poireaux, 50 brocolis rabe.

Legumineuses et produits cerealiers : 1 riz blanc, 2 riz brun, 3 quinoa, 4 couscous, 5 boulgour, 6 pâtes, 7 vermicelles, 8 semoule, 9 blé, 10 orge, 11 avoine, 
12 maïs, 13 millet, 14 sorgho, 15 sarrasin, 16 farine de blé, 17 farine de seigle, 18 farine de maïs, 19 farine de riz, 20 farine de pois chiches, 21 farine de lentilles, 
22 farine de noix de coco, 23 farine d'amande, 24 farine de souchet, 25 farine de lupin, 26 lentilles vertes, 27 lentilles corail, 28 pois cassés, 29 pois chiches, 30 fèves, 
31 haricots rouges, 32 haricots blancs, 33 haricots noirs, 34 soja, 35 tofu, 36 tempeh, 37 lait de soja, 38 lait d'amande, 39 lait de coco, 40 crème de soja, 41 yaourt de soja, 
42 fromage de soja, 43 fromage de chèvre, 44 fromage de brebis, 45 fromage de vache, 46 pain blanc, 47 pain complet, 48 pain aux céréales, 49 baguette, 50 biscottes.

Viandes, poissons et fruits de mer : 1 boeuf haché, 2 steak de boeuf, 3 viande de porc, 4 échine de porc, 5 côte de porc, 6 jambon blanc, 7 jambon cru, 8 bacon, 
9 saucisse de porc, 10 saucisson sec, 11 poulet, 12 blanc de poulet, 13 cuisse de poulet, 14 ailes de poulet, 15 foie de volaille, 16 canard, 17 magret de canard, 18 confit de canard, 
19 lapin, 20 sauté de lapin, 21 gigot d'agneau, 22 côte d'agneau, 23 agneau haché, 24 veau, 25 escalope de veau, 
26 osso bucco, 27 saumon, 28 filet de saumon, 29 truite, 30 cabillaud, 31 merlu, 32 sole, 33 thon, 34 moules, 35 huîtres, 36 palourdes, 37 coquilles Saint-Jacques, 
38 crevettes, 39 gambas, 40 homard, 41 crabe, 42 écrevisses, 43 escargots, 
44 foie gras, 45 magret de canard fumé, 46 poulet rôti, 47 jambon de Bayonne, 48 filet mignon de porc, 49 boudin noir, 50 saumon fumé

Produits sucrés et matières grasses : 1 sucre en poudre, 2 sucre roux, 3 cassonade, 4 miel, 5 sirop d'érable, 6 confiture, 7 pâte à tartiner, 8 chocolat noir, 
9 chocolat au lait, 10 chocolat blanc, 11 praliné, 12 crème fraîche, 13 crème liquide, 14 lait concentré sucré, 15 lait concentré non sucré, 16 lait en poudre, 17 beurre, 
18 margarine, 19 huile d'olive, 20 huile de tournesol, 21 huile de colza, 22 huile de noix, 23 vinaigre balsamique, 24 vinaigre de vin, 25 vinaigre blanc, 26 moutarde, 27 mayonnaise, 
28 ketchup, 29 sauce barbecue, 30 sauce soja, 31 sauce chili, 32 sauce béchamel, 33 sauce tomate, 34 sauce pesto, 35 yaourt nature, 36 yaourt aux fruits, 37 fromage blanc, 
38 crème dessert, 39 flan, 40 panna cotta, 41 mascarpone, 42 ricotta, 
43 cream cheese, 44 chantilly, 45 glace à la vanille, 46 glace au chocolat, 47 sorbet citron, 48 sorbet framboise, 49 pâte brisée, 50 pâte feuilletée.

'''

def all(ingredients):

  ingredients = MenageStr(ingredients)
  recette_decoupee = ingredients.clean_str

  recette_decoupee_dict = Isolation(recette_decoupee)
  recette_decoupee_dict = recette_decoupee_dict.rd_d


  def get_pair(prod, l_comp):

    # VARIABLES 
    coff_val = 1.0 
    pairs = [] 
    b_pair = "" 
    found = False
    stemmed_prod = return_stem(prod)

    # PROCESS
    while not pairs and coff_val >= 0 and found == False:

      pairs = difflib.get_close_matches(prod, l_comp, cutoff=coff_val)

      # SI ON A QQCH ON CHERCHE LE BON ITEM
      if pairs:

        # STEMMING + COMPARAISON
        pairs_set_score = []
        for i in range(len(pairs)):
          stemmed_item = return_stem(pairs[i])

          # LISTE DE SCORES DE RESSEMBLANCE
          pairs_set_score.append(len(set(stemmed_prod).intersection(stemmed_item)))

        # MEILLEURS ITEMS > 0
        if max(pairs_set_score) > 0:
          b_pair = pairs[pairs_set_score.index(max(pairs_set_score))]
          found = True
          break # On a mis found à True mais la boucle while ne se stoppe pas automatiquement, il faut break en plus

        # MOINS EXIGENT SI TOUT EST A 0
        else:
          coff_val -= 0.1
          pairs = []

      # MOINS EXIGENT SI AUCUN MATCH
      else:
        coff_val -= 0.1

    return b_pair


  # LISTE AVEC LES PAIRS ET LEURS INFOS
  pairs_infos = []
  for i in range(len(recette_decoupee_dict)):
    pairs_infos.append({recette_decoupee_dict[i]["Product"]:[], "Index":[]})

  def pairage():
    for i in range(len(recette_decoupee_dict)): 
      pair = get_pair(recette_decoupee_dict[i]["Product"], newFood)

      # SI ON TROUVE QQCH ON AJOUTE LES INFOS
      if pair:
        pairs_infos[i][recette_decoupee_dict[i]["Product"]].append(pair)
        pairs_infos[i]["Index"].append(newFood.index(pair))
      # N PARTOUT POUR DIRE QUE C'ETAIT PAS BON
      else:
        pairs_infos[i][recette_decoupee_dict[i]["Product"]].append("N")
        pairs_infos[i]["Index"].append("N")


  pairage()


  recette_decoupee_dict_fixed = Conversion(recette_decoupee_dict, dict_poids, pairs_infos)
  pairs_infos = recette_decoupee_dict_fixed.pairs_infos
  recette_decoupee_dict_fixed = recette_decoupee_dict_fixed.convert()

  resultat = Calcul(recette_decoupee_dict_fixed, nutriments, pairs_infos)

  return resultat.recette_table
