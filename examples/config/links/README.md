# Format

```yaml
module_name1:
  - instance_name1: yaml_name1
  - ...

module_name2:
  - instance_name2: yaml_name2
  - ...

LINK_SETTING: # some global setting attached to the link config
  - param_name1: value1
  - ...

PIPELINE_SETTING: # define the pipeline ("group name" is optional)
  - module1/name1/(group1) -> module2/name2/(tag2)
  - module2/name2/(group2) -> module3/name3/(tag3)
  - ...

# Function of "group" and "tag": 
# - (1) "group": Represent the output group name.
# - (2) "tag": Denote the different branch in the pipeline.
```