import os
import sys
path_name = os.path.abspath(
    os.path.join(__file__, "../../src")
)
print("path_name: ", path_name)
sys.path.insert(0, path_name)
import gedml

# generate method

"""
{
    "modules": {
        "class": {
            ...
        },
        "func": {
            ...
        }
    }
}
"""
def name_split(module):
    module_str = str(module)
    module_str = module_str.replace("'", "")
    module_str = module_str.replace("<", "")
    module_str = module_str.replace(">", "")
    output_tuple = module_str.split(" ")
    if len(output_tuple) != 2:
        return False, ""
    else:
        return output_tuple[0], output_tuple[1]

def get_output(module):
    output_dict = {"class": {}, "func": {}}
    attr_dict = module.__dict__
    for k, v in attr_dict.items():
        type_name, description = name_split(v)
        if not type_name:
            continue
        
        # if class
        if type_name == "class":
            name = description.split(".")[-1]
            output_dict["class"][name] = description

        # if func
        elif type_name == "":
            pass
        
        # if module
        elif type_name == "module":
            pass
    return output_dict

def write_rst(
    module, 
    module_description, 
    sub_dict, 
    path=os.path.abspath(os.path.join(
        __file__, "../source/"
    ))
):
    h1 = "##################################"
    h2 = "**********************************"
    h3 = "++++++++++++++++++++++++++++++++++"
    module_utils = "\t:members:\n\n"
    class_utils = "\t:members:\n\t:show-inheritance:\n\n"
    file_name = module + ".rst"
    lines_list = [module+"\n"]
    lines_list.append(h1+"\n\n")
    # append module
    lines_list.append(".. automodule:: {}\n{}\n".format(
        module_description, module_utils
    ))
    # append class
    lines_list.append("Class\n"+h2+"\n\n")
    for k, v in sub_dict["class"].items():
        lines_list.append(k+"\n"+h3+"\n")
        lines_list.append(".. autoclass:: {}\n{}\n".format(
            v, class_utils
        ))
    # append func
    pass

    with open(path+file_name, mode="w") as f:
        f.writelines(lines_list)

# hyper-parameters

to_parse_dict = {
    "core": ["collectors", "losses", "datasets", "evaluators", 
            "metrics", "models", "modules", "samplers", 
            "selectors", "transforms"],
    "launcher": ["creators", "managers", "testers", "trainers"]
}


for base_module, to_generate_list in to_parse_dict.items():
    output_dict = {}
    base_module = getattr(gedml, base_module)
    for name in to_generate_list:
        module = getattr(base_module, name)
        output_dict[name] = get_output(module)
    for module, sub_dict in output_dict.items():
        module_description = getattr(base_module, module).__name__
        write_rst(module, module_description, sub_dict)
    pass