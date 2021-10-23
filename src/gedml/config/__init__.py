"""
This module is designed for storing all the parameters. 
Most of the config data is stored in yaml files. For more information about `yaml`, please refer to `YAML <https://yaml.org/>`_.
The following sections explain the usage of this module from three aspects:

1. code structure
2. custom setting
3. demo

1 Code Structure
**********************

===============================================

- `param <https://github.com/zbr17/GeDML/tree/v1.0.0/src/config/param>`_
    - models
    - losses
    - ...
- `setting <https://github.com/zbr17/GeDML/tree/v1.0.0/src/config/setting>`_
    - core_setting.py
    - launcher_setting.py
    - recorder_setting.py
- `assert.yaml <https://github.com/zbr17/GeDML/blob/v1.0.0/src/config/assert.yaml>`_
- `convert.yaml <https://github.com/zbr17/GeDML/blob/v1.0.0/src/config/convert.yaml>`_
- `link.yaml <https://github.com/zbr17/GeDML/blob/v1.0.0/src/config/link.yaml>`_

===============================================

1.1 Global Yaml Setting Files
++++++++++++++++++++++++++++++++++++++++++++++

1. ``link.yaml``. Assemble the module parameters which you need to use. This yaml file has a strict writing format, please refer to :ref:`config-format`.

2. ``convert.yaml``. After all the parameters are integrated, we may want to modify some of there parameters. To conveniently modify this parameters, this yaml file define the mapping dict.

3. ``assert.yaml``. **(todo)** Sets the self-check options for intialization, because there are dependencies between algorithm modules. For example, `TripletLoss` can only be used with Euclidean distance. In addition, there is a matching relation between the parameters of different modules. For example, `Resnet50` must be matched with :math:`feature\_size=2048`.

1.2 ``params`` Folder
++++++++++++++++++++++

Store commonly used parameters include ``models``, ``collectors``, ``datasets``, ``evaluators``, ``gradclipper``, ``losses``, ``managers``, ... Module `launcher.creators.ConfigHandler <https://github.com/zbr17/GeDML/blob/v1.0.0/src/launcher/creators/config_handler.py>`_ will fetch specific parameters from ``params`` folder according to a ``link.yaml`` file and merge all the ``.yaml`` files into a whole **Python Dictionay**.

1.3 ``setting`` Folder
++++++++++++++++++++++

There are three ``.py`` files which define the protocols: 

- ``core_setting.py``. Define some strings which may be used by ``core`` module.
- ``launcher_setting.py``. Define some strings or mapping function which may be used by ``launcer`` module and key variables which control the whole framework (e.g. intialization order list)
- ``recorder_setting.py``. Define some strings which may be used by ``recorder`` module (if any module wants to add ``recorder`` property, it must be define according to ``.py`` file).

.. _config-format:

2 Custom Setting
**********************

If you want to add more settings, there are strict formats that you must follow.

2.1 ``link.yaml`` format
+++++++++++++++++++++++++++++

**Format**:

.. code:: yaml

    module_name1:
      - instance_name1: config_file_name1.yaml
      - instance_name2: config_file_name2.yaml
    
    module_name2:
      - ...
    
    ...

    LINK_SETTING:
      - variable_name: value

- ``module_name``: This part specifies the parameter files to be merged.
- ``LINK_SETTING``: Addressed to some methods, ``LINK_SETTING`` provides convenience for modifying parameters from ``link\_\*.yaml``.

.. note::
    Although there is a strict order of intialization which is defined in `config.setting.launcher_setting <https://github.com/zbr17/GeDML/blob/v1.0.0/src/config/setting/launcher_setting.py>`_, the order of modules in ``link.yaml`` is not required. 

2.2 ``convert.yaml`` format
+++++++++++++++++++++++++++++

**Format**:

.. code:: yaml

    variable_name1:
      - module_name1/instance_name1-param_name1
      - module_name2/instance_name2-param_name2
    variable_name2:
      ...

This file provides a convenient mapping from a **simple string** to **complex parameters paths**. 

2.3 **assert.yaml** format
+++++++++++++++++++++++++++++

**(todo)**

2.4 **params** format
+++++++++++++++++++++++++++++

Note:
    Because the names of sub-folder in "params" folder strictly corresponds to the names of certain modules, DO NOT modify the names of these folders.

**Format**::

    classname:
      params:
        param_name1: value1
        param_name2: value2
        ...
      INITIATE: initiate_method_name

There are two keywords:

- **params**: Set all parameters under this keyword.
- **INITIATE**: Set special initialization method (default **DO NOTHING**). Each module corresponds to a **creator** initializer which defines "How this module can be initiated". For example, models' creator `modelsCreator <https://github.com/zbr17/GeDML/blob/v1.0.0/src/launcher/creators/factories/models_creator.py>`_ has defined FOUR initialization methods:
    - (default do nothing)
    - **delete_last_linear**
    - **freeze_all**
    - **freeze_except_last_linear**

And there are two kinds of parameters:

- **Hyperparameters**. These parameters can be defined before training started and the type of them may be **int, float, list, dict, etc**. 
- **Instances**. These parameters are Python instance which must be created during training. For example, **schedulers** have "optimizer" parameter to be passed. Therefore, three types of "fetching" methods are defined:
    - **~~_PASS_WITH_NAMED_MEMBER_**. Pass all instances (a dictionary).
    - **~~_SEARCH_WITH_SAME_NAME_**. Pass instances with the same name.
    - **~~_SEARCH_WITH_TARGET_NAME_**. Pass instances according to the specific target name.

Here are the formats::

    param_name: 
      - ~~_PASS_WITH_NAMED_MEMBER_
      - (none)
      - (none)

    param_name:
      - ~~_SEARCH_WITH_SAME_NAME_
      - module_name
      - (none)

    param_name:
      - ~~_SEARCH_WITH_TARGET_NAME_
      - module_name
      - instance_name

3 Demo
**********************

Some **demo** configs have been prepared in `demo <https://github.com/zbr17/GeDML/tree/v1.0.0/demo>`_. A **HDML** config demo is displayed below:

Example::

    metrics:
      - default: euclid.yaml

    collectors:
      - default: HDMLCollector.yaml

    selectors:
      - default: DenseTripletSelector.yaml

    losses:
      - default: TripletLoss.yaml

    models:
      - trunk: googlenet.yaml
      - embedder: hdml_embedder.yaml
      - generator: hdml_generator.yaml
      - embedder_recon: hdml_embedder.yaml
      - classifier: hdml_classifier.yaml

    evaluators:
      - default: Calculator.yaml

    optimizers:
      - trunk: RMSprop_trunk.yaml
      - embedder: RMSprop_embedder.yaml
      - generator: Adam_hdml_generator.yaml
      - embedder_recon: RMSprop_embedder.yaml
      - classifier: Adam_hdml_classifier.yaml

    schedulers:
      - trunk: on_plateau.yaml
      - embedder: on_plateau.yaml
      - generator: on_plateau.yaml
      - embedder_recon: on_plateau.yaml
      - classifier: on_plateau.yaml

    gradclipper:
      - trunk: gradclipper.yaml
      - embedder: gradclipper.yaml
      - generator: gradclipper.yaml
      - embedder_recon: gradclipper.yaml
      - classifier: gradclipper.yaml

    transforms:
      - train: train_transforms.yaml
      - test: eval_transforms.yaml

    datasets:
      - train: cub200_train.yaml
      - test: cub200_test.yaml

    recorders:
      - default: base_recorder.yaml

    trainers: 
      - default: BaseTrainer.yaml

    testers:
      - default: BaseTester.yaml

    managers:
      - default: BaseManager.yaml

    LINK_SETTING:
      to_device_list: [models]
      to_wrap_list: [models]


"""