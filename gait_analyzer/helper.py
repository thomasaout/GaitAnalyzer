import inspect


def helper(class_instance):
    """
    This function prints the available functions of a class along with their inputs, outputs, and descriptions.

    Parameters
    ----------
    class_instance: class
        The class instance for which the available functions should be printed
    """
    # Checks
    if not callable(class_instance):
        raise ValueError("The input must be a class instance.")

    # Print the signature of the class
    print(f"{class_instance}{inspect.signature(class_instance)}")
    print("The following functions are available, along with their inputs, outputs, and descriptions:")
    # Print the available functions
    for method in dir(class_instance):
        if callable(getattr(class_instance, method)) and not method.startswith("__"):
            func = getattr(class_instance, method)
            sig = inspect.signature(func)
            return_annotation = sig.return_annotation
            if return_annotation is inspect._empty:
                return_annotation = "None"
            initial_doc = inspect.getdoc(func)
            if initial_doc is None:
                doc = "  No documentation available."
            else:
                initial_doc = initial_doc.split("\n")
                doc = ""
                for line in initial_doc:
                    if line.strip() != "":
                        doc += "  " + line.strip() + "\n"
            print(f"\n- {method}{sig} -> {return_annotation}")
            print(f"{doc}\n")


