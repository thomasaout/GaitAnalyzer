
import opensim as osim
# This code necessitates to install the OpenSim Python API. You can do it by running the following command:
# conda install opensim-org::opensim
# However, please note that your python version must be python=3.11


def convert_c3d_files(file_path: str):

    c3d_adapter = osim.C3DFileAdapter()
    sto_adapter = osim.STOFileAdapter()

    tables = c3d_adapter.read(file_path)
    markers_table = c3d_adapter.getMarkersTable(tables)
    sto_adapter.write(markers_table.flatten(), file_path.replace(".c3d", ".trc"))
    # forces_table = c3d_adapter.getForcesTable(tables)
    # sto_adapter.write(forces_table.flatten(), file_path.replace(".c3d", ".sto"))


# Example of how to use the function
convert_c3d_files("../data/AOT_01/AOT_01_ManipStim_Statique.c3d")





