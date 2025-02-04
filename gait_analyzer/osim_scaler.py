import os
import fnmatch
import pandas as pd
from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog
import ezc3d
import opensim
from xml.etree import ElementTree as ET


# Étape 1 : Sélection du dossier participant
def select_participant_folder():
    app = QApplication.instance() or QApplication([])
    folder_path = QFileDialog.getExistingDirectory(None, "Sélectionnez le dossier du participant")
    if not folder_path:
        raise FileNotFoundError("Aucun dossier sélectionné.")
    return folder_path


# Étape 2 : Saisie du nom du participant
def get_participant_name():
    app = QApplication.instance() or QApplication([])
    participant_name, ok = QInputDialog.getText(None, "Nom du participant", "Entrez le nom du participant :")
    if not ok or not participant_name:
        raise ValueError("Aucun nom de participant saisi.")
    return participant_name


# Étape 3 : Chargement des données anthropométriques depuis Excel
def load_anthropometric_data(excel_path, participant_name):
    df = pd.read_excel(excel_path, sheet_name="Resume")
    participant_row = df[df.iloc[:, 2] == participant_name]  # Colonne 3
    if participant_row.empty:
        raise ValueError("Participant non trouvé dans le fichier Excel.")

    data = {
        "Sexe": participant_row.iloc[0, 4],  # Colonne 5
        "Age": participant_row.iloc[0, 5],  # Colonne 6
        "Taille": participant_row.iloc[0, 6],  # Colonne 7
        "Masse": participant_row.iloc[0, 7],  # Colonne 8
        "Vitesse": participant_row.iloc[0, 8]  # Colonne 9
    }
    return data


# Étape 4 : Recherche du fichier C3D statique
def find_static_c3d(participant_folder):
    for file in os.listdir(participant_folder):
        if fnmatch.fnmatch(file.lower(), "*statique.c3d"):  # Corrigé : "tatique" -> "statique"
            return os.path.join(participant_folder, file)
    raise FileNotFoundError("❌ Aucun fichier C3D statique trouvé dans le dossier du participant.")

# Étape 5 : Création des dossiers output et result
def create_output_folders(participant_folder, participant_name):
    output_path = os.path.join(participant_folder, "output")
    model_path = os.path.join(output_path, f"Model_{participant_name}")
    os.makedirs(model_path, exist_ok=True)
    return model_path


# Étape 6 : Création du fichier TRC
def create_trc_from_c3d(c3d_file, trc_file):
    c3d = ezc3d.c3d(c3d_file)
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    frame_rate = c3d["header"]["points"]["frame_rate"]
    marker_data = c3d["data"]["points"][:3, :, :] / 1000  # Conversion en mètres

    marker_data = marker_data[[0, 2, 1], :, :]  # Réorientation des axes
    marker_data[2, :, :] *= -1  # Inversion de l'axe Z pour OpenSim

    with open(trc_file, "w") as f:
        f.write("PathFileType\t4\t(X/Y/Z)\t{}\n".format(trc_file))
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write("{:.2f}\t{:.2f}\t{}\t{}\tm\t{:.2f}\t{}\t{}\n".format(
            frame_rate, frame_rate, c3d["header"]["points"]["last_frame"], len(labels),
            frame_rate, c3d["header"]["points"]["first_frame"], c3d["header"]["points"]["last_frame"]
        ))
        f.write("Frame#\tTime\t" + "\t".join(labels) + "\n")
        f.write("\t\t" + "\t".join([f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(len(labels))]) + "\n")

        for frame in range(marker_data.shape[2]):
            time = frame / frame_rate
            frame_data = [f"{frame + 1}\t{time:.5f}"]
            for marker_idx in range(len(labels)):
                pos = marker_data[:, marker_idx, frame]
                frame_data.extend([f"{pos[0]:.5f}", f"{pos[1]:.5f}", f"{pos[2]:.5f}"])
            f.write("\t".join(frame_data) + "\n")
    print(f"✅ Fichier TRC généré : {trc_file}")


# Étape 7 : Modification du fichier XML
def modify_xml(xml_path, participant_name, mass, trc_path, model_path):
    """
    Modifie un fichier XML de configuration pour le Scale Tool et génère un nouveau fichier.
    :param xml_path: Chemin du fichier XML de base
    :param participant_name: Nom du participant
    :param mass: Masse du participant
    :param trc_path: Chemin du fichier TRC
    :param model_path: Dossier de sortie pour le fichier modifié
    :return: Chemin absolu du nouveau fichier XML
    """
    # Définir le chemin du modèle scalé et du fichier XML modifié
    scaled_model_path = os.path.join(model_path, f"{participant_name}_scaled.osim")
    new_xml_path = os.path.join(model_path, f"{participant_name}_setup.xml")

    # Charger et modifier le fichier XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for elem in root.iter():
        if elem.tag == "model_file":
            elem.text = "/Users/thomasaout/Documents/Python/manip_4/49markers.osim"
        if elem.tag == "output_model_file":
            elem.text = scaled_model_path
        if elem.tag == "mass":
            elem.text = f"{mass}"
        if elem.tag == "marker_file":
            elem.text = os.path.abspath(trc_path)  # Chemin absolu propre du fichier TRC

    # Sauvegarder le nouveau fichier XML
    tree.write(new_xml_path)

    # Chemin absolu du fichier XML créé
    #chemin = os.path.abspath(new_xml_path)
    chemin = new_xml_path

    # Afficher le chemin dans la commande
    print(f"✅ Nouveau fichier XML créé : {chemin}")

    # Retourner le chemin absolu
    return chemin

# Étape 8 : Exécution de Scale Tool
def run_scale_tool(chemin):
    print(f"📄 Chargement du fichier XML : {chemin}")
    tool = opensim.ScaleTool(chemin)
    tool.run()

    try:
        tool.run()
        print("✅ Modèle mis à l'échelle généré avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution de ScaleTool : {e}")


# Pipeline principal
if __name__ == "__main__":
    # participant_folder = select_participant_folder()
    # participant_name = get_participant_name()

    participant_name = "AOT"
    model_path = "49markers.osim"
    static_c3d_path = "ManipStim_AOT_01_Statique.c3d"
    trc_path = "ManipStim_AOT_01_static.trc"

    anthropometric_data = load_anthropometric_data("/Users/thomasaout/Desktop/etude_4/Manip4.xlsx", participant_name)
    # static_c3d_path = find_static_c3d(participant_folder)
    # model_path = create_output_folders(participant_folder, participant_name)
    # model_path = create_output_folders(participant_folder, participant_name)

    # trc_path = os.path.join(model_path, f"{participant_name}_static.trc")
    create_trc_from_c3d(static_c3d_path, trc_path)

    xml_template_path = "/Users/thomasaout/Documents/Python/manip_4/setup.xml"
    # new_xml_path = modify_xml(xml_template_path, participant_name, anthropometric_data["Masse"], trc_path, model_path)
    # chemin = os.path.abspath(new_xml_path)
    chemin = "AOT_setup.xml"

    run_scale_tool(chemin)  # Exécuter directement le fichier XML modifié