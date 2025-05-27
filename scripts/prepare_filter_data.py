import pandas as pd
import numpy as np


"""
    Prépare un dataset F1 enrichi à partir de plusieurs fichiers sources :
    - Résultats de course
    - Pilotes, écuries, statuts, etc.

    Le fichier produit contient :
    - Des variables cibles pour la classification et la régression
    - Des caractéristiques enrichies (nationalités, circuit_type, home_race)
    - Une gestion des valeurs manquantes adaptées au contexte F1

    Retourne un DataFrame prêt à l'analyse.
"""


def data_preparation():
    
    # Chargement des données
    results = pd.read_csv("data/results.csv", na_values="\\N")
    status = pd.read_csv("data/status.csv", na_values="\\N")
    races = pd.read_csv("data/races.csv", na_values="\\N")
    constructors = pd.read_csv("data/constructors.csv", na_values="\\N")
    drivers = pd.read_csv("data/drivers.csv", na_values="\\N")

    # Fusion des tables principales
    data = results.merge(races[["raceId", "year", "name"]], on="raceId", how="left")
    data.rename(columns={"name": "races_name"}, inplace=True)
    data = data.merge(status, on="statusId", how="left")

    # Ajout des infos sur les pilotes

    # Nettoyage et séparation des nationalités
    def split_and_clean_nationality(nat):
        if pd.isna(nat):
            return [np.nan, np.nan]
        nat = nat.strip().title()
        corrections = {
            "Argentinian": "Argentine",
            "Argentinian ": "Argentine",
            "Argentine-Italian": "Argentine-Italian",
            "American-Italian": "American-Italian",
            "East German": "German",
            "Rhodesian": "South African",
            "New Zealander": "New Zealander",
        }
        nat = corrections.get(nat, nat)
        parts = [n.strip().title() for n in nat.split("-")]
        if len(parts) == 1:
            return [parts[0], np.nan]
        return parts[:2]

    # Appliquer au DataFrame drivers
    drivers[["driver_nationality_1", "driver_nationality_2"]] = drivers["nationality"].apply(split_and_clean_nationality).apply(pd.Series)

    # Fusionner les données pilotes dans 'data'
    data = data.merge(
        drivers[["driverId", "dob", "driver_nationality_1", "driver_nationality_2", "forename", "surname"]],
        on="driverId", how="left"
    )

    # Âge et nom du pilote
    data["driver_age"] = data["year"] - pd.to_datetime(data["dob"], errors="coerce").dt.year
    data["driver_name"] = data["forename"] + " " + data["surname"]
    data = data.drop(columns=["forename", "surname"])

    # Ajout des infos sur les constructeurs
    data = data.merge(constructors[["constructorId", "name", "nationality"]], on="constructorId", how="left")
    data.rename(columns={"name": "constructor_name", "nationality": "constructor_nationality"}, inplace=True)

    

    # ---------------------------
    # Cibles de classification
    # ---------------------------


    # 1. Le pilote a-t-il terminé la course ?
    data["target_finish_race"] = data["status"].str.contains("Finished", case=False, na=False).astype(int)
    # 2. Le pilote a-t-il fini sur le podium ?
    data["target_podium"] = data["positionOrder"].isin([1, 2, 3]).astype(int)
    # 3. Le pilote est-il parti dans le top 10 ?
    data["target_top10"] = (data["grid"] <= 10).astype(int)
    # 4. Le pilote a-t-il gagné des places ?
    data["target_gain_position"] = (data["positionOrder"] < data["grid"]).astype(int)
    # 5. Le pilote a-t-il battu son coéquipier ?
    best_teammate = data.groupby(["raceId", "constructorId"])["positionOrder"].transform("min")
    data["target_beat_teammate"] = (data["positionOrder"] == best_teammate).astype(int)
 

    # ---------------------------
    # Cibles de régression
    # ---------------------------

    # 1. Nombre de tours complétés par le pilote
    data["laps_completed"] = data["laps"]

    # 2. Temps total de course (exprimé en secondes à partir des millisecondes)
    if "milliseconds" in data.columns:
        data["time_total_seconds"] = data["milliseconds"] / 1000
    else:
        data["time_total_seconds"] = np.nan  # au cas où la colonne est absente

    # 3. Position finale du pilote
    data["final_position"] = data["positionOrder"]

    # 4. Nombre de places gagnées en course
    data["gained_positions"] = data["grid"] - data["positionOrder"]

    # 5. Nombre de points marqués par une écurie (somme par année + écurie)
    data["team_points"] = data.groupby(["raceId", "constructor_name"])["points"].transform("sum")


    # ---------------------------
    # Traitement des données pilotes OUT
    # ---------------------------


    # Crée une colonne finished_race pour plus de clarté
    data["finished_race"] = data["status"].str.contains("Finished", case=False, na=False).astype(int)

    # Si le pilote n'a pas fini, fastestLapSpeed, fastestLapTime et rank doivent être NaN ou neutres
    if "fastestLapSpeed" in data.columns:
        data.loc[data["finished_race"] == 0, "fastestLapSpeed"] = np.nan
        data["fastestLapSpeed"] = data["fastestLapSpeed"].fillna(0)

    if "rank" in data.columns:
        data.loc[data["finished_race"] == 0, "rank"] = np.nan
        data["rank"] = data["rank"].fillna(100)  # 100 = très mauvais classement fictif

    if "fastestLapTime" in data.columns:
        data["fastestLapTime_missing"] = data["fastestLapTime"].isna().astype(int)
        data["fastestLapTime"] = data["fastestLapTime"].fillna("0:00.000")


    # ---------------------------
    # Nettoyage final
    # ---------------------------


    # Liste des colonnes inutiles
    columns_to_drop = [
        "resultId", "raceId", "driverId", "constructorId", "number",
        "positionText", "statusId", "dob"  # dob n'est plus utile une fois driver_age calculé
    ]

    # Suppression des colonnes inutiles
    data = data.drop(columns=columns_to_drop)

    # Création d'une variable binaire indiquant si le pilote n'a pas réalisé de meilleur tour (rank = 100)
    data["rank_missing"] = (data["rank"] == 100).astype(int)


    # ---------------------------
    # Détection des courses à domicile
    # ---------------------------

    # Mapping manuel entre races_name et nationalité du pays hôte
    gp_to_nationality = {
        "70th Anniversary Grand Prix": "British",
        "Abu Dhabi Grand Prix": "Emirati",
        "Argentine Grand Prix": "Argentine",
        "Australian Grand Prix": "Australian",
        "Austrian Grand Prix": "Austrian",
        "Azerbaijan Grand Prix": "Azerbaijani",
        "Bahrain Grand Prix": "Bahraini",
        "Belgian Grand Prix": "Belgian",
        "Brazilian Grand Prix": "Brazilian",
        "British Grand Prix": "British",
        "Caesars Palace Grand Prix": "American",
        "Canadian Grand Prix": "Canadian",
        "Chinese Grand Prix": "Chinese",
        "Dallas Grand Prix": "American",
        "Detroit Grand Prix": "American",
        "Dutch Grand Prix": "Dutch",
        "Eifel Grand Prix": "Dutch",
        "Emilia Romagna Grand Prix": "Italian",
        "European Grand Prix": None,
        "French Grand Prix": "French",
        "German Grand Prix": "German",
        "Hungarian Grand Prix": "Hungarian",
        "Indian Grand Prix": "Indian",
        "Indianapolis 500": "American",
        "Italian Grand Prix": "Italian",
        "Japanese Grand Prix": "Japanese",
        "Korean Grand Prix": "South Korean",
        "Las Vegas Grand Prix": "American",
        "Luxembourg Grand Prix": "Luxembourgish",
        "Malaysian Grand Prix": "Malaysian",
        "Mexican Grand Prix": "Mexican",
        "Mexico City Grand Prix": "Mexican",
        "Miami Grand Prix": "American",
        "Monaco Grand Prix": "Monegasque",
        "Moroccan Grand Prix": "Moroccan",
        "Pacific Grand Prix": "Japanese",
        "Pescara Grand Prix": "Italian",
        "Portuguese Grand Prix": "Portuguese",
        "Qatar Grand Prix": "Qatari",
        "Russian Grand Prix": "Russian",
        "Sakhir Grand Prix": "Bahraini",
        "San Marino Grand Prix": "Italian",
        "Saudi Arabian Grand Prix": "Saudi",
        "Singapore Grand Prix": "Singaporean",
        "South African Grand Prix": "South African",
        "Spanish Grand Prix": "Spanish",
        "Styrian Grand Prix": "Austrian",
        "Swedish Grand Prix": "Swedish",
        "Swiss Grand Prix": "Swiss",
        "São Paulo Grand Prix": "Brazilian",
        "Turkish Grand Prix": "Turkish",
        "Tuscan Grand Prix": "Italian",
        "United States Grand Prix": "American",
        "United States Grand Prix West": "American"
    }

    # Créer une colonne avec la nationalité du pays organisateur
    data["gp_nationality"] = data["races_name"].map(gp_to_nationality)

    # Comparer avec la nationalité du pilote
    data["is_home_race"] = ((data["gp_nationality"] == data["driver_nationality_1"]) | (data["gp_nationality"] == data["driver_nationality_2"])).astype(int)


    # ---------------------------
    # Ajout de circuit_type (typologie du tracé)
    # ---------------------------
    circuit_type_map = {
    "Monaco": "urbain", "Marina Bay": "urbain", "Miami": "urbain", "Las Vegas": "urbain", "Djeddah": "urbain", "Baku": "urbain",
    "Suzuka": "technique", "Hungary": "technique", "Zandvoort": "technique", "Imola": "classique", "Barcelona": "équilibré",
    "Silverstone": "rapide", "Spa": "très rapide", "Monza": "ultra-rapide", "Mexico": "altitude", "Austin": "moderne",
    "Sakhir": "rapide", "Qatar": "moderne", "Shanghai": "moderne", "Yas Marina": "moderne"
    }

    def get_circuit_type(race_name):
        for key in circuit_type_map:
            if key.lower() in race_name.lower():
                return circuit_type_map[key]
        return "inconnu"

    data["circuit_type"] = data["races_name"].apply(get_circuit_type)


    # Export final
    data.to_csv("data/data_filter.csv", index=False)

    return data

# Test rapide
if __name__ == "__main__":
    df = data_preparation()
    print(df[[
        "positionOrder", "grid", "status",
        "driver_nationality_1", "driver_nationality_2", "driver_age", "driver_name",
        "constructor_name", "constructor_nationality",
        "target_finish_race", "target_podium", "target_top10",
        "target_gain_position", "target_beat_teammate", "rank",
        "rank_missing","is_home_race", "circuit_type"
    ]].head())

    print(df[[
        "laps_completed", "time_total_seconds", "final_position",
        "gained_positions", "team_points"
    ]].describe())

