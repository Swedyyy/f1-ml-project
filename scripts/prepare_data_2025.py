import pandas as pd
import numpy as np
import os

"""
Génère le dataset simulé de la saison 2025 de F1 à partir :
    - des pilotes 2025,
    - des caractéristiques circuits,
    - et des performances écuries de 2024.

    Applique également les règles de changement d'écurie et d'alternance pilote.
    Retourne un DataFrame prêt à l'usage pour la prédiction.
"""

def prepare_data_2025():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    drivers_path = os.path.join(root_dir, "data", "season2025", "drivers_2025.csv")
    hist_path = os.path.join(root_dir, "data", "data_filter.csv")
    output_path = os.path.join(root_dir, "data", "season2025", "data_2025.csv")

    drivers_2025 = pd.read_csv(drivers_path)
    df_hist = pd.read_csv(hist_path)

    # Harmonisation des noms d'écuries entre historiques et saison 2025
    replace_team_names = {
    "Red Bull Racing": "Red Bull",
    }

    # Remplacer les noms dans drivers_2025 AVANT génération des données
    drivers_2025["Team"] = drivers_2025["Team"].replace(replace_team_names)
    df_hist["constructor_name"] = df_hist["constructor_name"].replace(replace_team_names)


    # Calcul des points des écuries en 2024
    df_2024 = df_hist[df_hist["year"] == 2024]
    team_points = df_2024.groupby("constructor_name")["points"].sum().reset_index()
    team_points.rename(columns={"points": "team_points_last_y"}, inplace=True)

    circuits_2025 = [
        "Australie (Melbourne)", "Chine (Shanghai)", "Japon (Suzuka)", "Bahreïn (Sakhir)",
        "Arabie Saoudite (Djeddah)", "Miami", "Émilie-Romagne (Imola)", "Monaco (Monte-Carlo)",
        "Espagne (Barcelone)", "Canada (Montréal)", "Autriche (Spielberg)", "Angleterre (Silverstone)",
        "Belgique (Spa-Francorchamps)", "Hongrie (Budapest)", "Pays-Bas (Zandvoort)", "Italie (Monza)",
        "Azerbaïdjan (Bakou)", "Singapour (Marina Bay)", "États-Unis (Austin)", "Mexique (Mexico City)",
        "Brésil (São Paulo)", "Las Vegas", "Qatar (Losail)", "Abou Dhabi (Yas Marina)"
    ]

    # Dictionnaires de mapping
    type_circuit_map = {
        "Melbourne": "semi-urbain", "Shanghai": "moderne", "Suzuka": "technique", "Sakhir": "rapide",
        "Djeddah": "urbain", "Miami": "urbain", "Imola": "classique", "Monte-Carlo": "urbain",
        "Barcelone": "équilibré", "Montréal": "semi-urbain", "Spielberg": "rapide", "Silverstone": "rapide",
        "Spa-Francorchamps": "très rapide", "Budapest": "technique", "Zandvoort": "rapide", "Monza": "ultra-rapide",
        "Bakou": "urbain", "Marina Bay": "urbain", "Austin": "moderne", "Mexico City": "altitude",
        "São Paulo": "rapide", "Las Vegas": "urbain", "Losail": "moderne", "Yas Marina": "moderne"
    }

    laps_mean_time_map = {
        "Melbourne": 88.3, "Shanghai": 96.4, "Suzuka": 90.1, "Sakhir": 87.6, "Djeddah": 82.7,
        "Miami": 88.2, "Imola": 85.3, "Monte-Carlo": 71.3, "Barcelone": 84.9, "Montréal": 75.3,
        "Spielberg": 64.3, "Silverstone": 86.1, "Spa-Francorchamps": 105.0, "Budapest": 77.4,
        "Zandvoort": 72.7, "Monza": 63.7, "Bakou": 100.2, "Marina Bay": 110.3, "Austin": 93.4,
        "Mexico City": 77.3, "São Paulo": 71.9, "Las Vegas": 87.5, "Losail": 88.7, "Yas Marina": 89.4
    }

    weather_profile_map = {
        "Melbourne": "variable", "Shanghai": "pluvieux", "Suzuka": "pluvieux", "Sakhir": "sec",
        "Djeddah": "sec", "Miami": "humide", "Imola": "variable", "Monte-Carlo": "sec",
        "Barcelone": "sec", "Montréal": "variable", "Spielberg": "pluvieux", "Silverstone": "pluvieux",
        "Spa-Francorchamps": "très pluvieux", "Budapest": "sec", "Zandvoort": "variable", "Monza": "sec",
        "Bakou": "sec", "Marina Bay": "humide", "Austin": "sec", "Mexico City": "sec",
        "São Paulo": "variable", "Las Vegas": "sec", "Losail": "sec", "Yas Marina": "sec"
    }

    home_race_map = {
        "français": "Monaco", "monégasque": "Monaco", "britannique": "Silverstone",
        "espagnol": "Barcelone", "italien": "Monza", "néerlandais": "Zandvoort",
        "allemand": "Hockenheim", "japonais": "Suzuka", "chinois": "Shanghai",
        "canadien": "Montréal", "brésilien": "São Paulo", "mexicain": "Mexico City",
        "argentin": "", "australien": "Melbourne", "américain": "Austin"
    }

    # Construction du dataset brut
    data_2025 = []
    for circuit in circuits_2025:
        for _, driver in drivers_2025.iterrows():
            circuit_key = next((k for k in type_circuit_map if k in circuit), None)
            nationality = driver["driver_nationality_1"].lower()
            home_gp = home_race_map.get(nationality, "")
            is_home = int(home_gp.lower() in circuit.lower())

            data_2025.append({
                "driver_name": driver["Full Name"],
                "constructor_name": driver["Team"],
                "driver_nationality_1": driver["driver_nationality_1"],
                "driver_age": driver["driver_age"],
                "circuit_name": circuit,
                "circuit_type": type_circuit_map.get(circuit_key, "inconnu"),
                "laps_mean_time": laps_mean_time_map.get(circuit_key, np.nan),
                "weather_profile": weather_profile_map.get(circuit_key, "inconnu"),
                "is_home_race": is_home
            })

    df_data_2025 = pd.DataFrame(data_2025)

    # Application des changements d’écurie à partir de Suzuka
    gp_changement = "Japon (Suzuka)"
    courses_apres_japon = circuits_2025[circuits_2025.index(gp_changement):]

    df_data_2025.loc[
        (df_data_2025["driver_name"] == "Yuki Tsunoda") &
        (df_data_2025["circuit_name"].isin(courses_apres_japon)),
        "constructor_name"
    ] = "Red Bull"

    df_data_2025.loc[
        (df_data_2025["driver_name"] == "Liam Lawson") &
        (df_data_2025["circuit_name"].isin(courses_apres_japon)),
        "constructor_name"
    ] = "Racing Bulls"

    # Suppression de l’alternance Alpine
    courses_doohan = ["Australie (Melbourne)", "Chine (Shanghai)", "Japon (Suzuka)", "Bahreïn (Sakhir)", "Arabie Saoudite (Djeddah)", "Miami"]
    courses_colapinto = circuits_2025[circuits_2025.index("Émilie-Romagne (Imola)"):]

    df_data_2025 = df_data_2025[~(
        (df_data_2025["driver_name"] == "Franco Colapinto") &
        (df_data_2025["circuit_name"].isin(courses_doohan))
    )]

    df_data_2025 = df_data_2025[~(
        (df_data_2025["driver_name"] == "Jack Doohan") &
        (df_data_2025["circuit_name"].isin(courses_colapinto))
    )]

    # Ajout des points des écuries
    df_data_2025 = df_data_2025.merge(team_points, on="constructor_name", how="left")
    df_data_2025["team_points_last_y"] = df_data_2025["team_points_last_y"].fillna(0.0)

    # Sauvegarde
    df_data_2025.to_csv(output_path, index=False)
    print(f" Dataset 2025 sauvegardé avec succès : {df_data_2025.shape[0]} lignes.")
    return df_data_2025

if __name__ == "__main__":
    df = prepare_data_2025()
    print(df.head())
