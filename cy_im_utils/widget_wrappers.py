from pathlib import Path
from ipywidgets import widgets,Layout


def file_select_widget(directory=Path("."),
        glob_regex: str = "",
        width: str = "90%",
        height: str = "200px",
        desc: str = "select file"
        ):
    """ This funciton is used in the context of a jupyter notebook to provide a
    dropdown menu of *.ini files down one directory...
    """
    file_set = sorted(list(directory.glob(glob_regex)))
    if len(file_set) == 0:
        print("warning no files found")
    file_names = [elem.name for elem in file_set]
    data_set_select = widgets.Select(options=file_names,
                                     layout=Layout(width=width, height=height),
                                     description=desc
                                     )
    display(data_set_select)
    return data_set_select
