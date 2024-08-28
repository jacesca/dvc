import yaml
from pprint import pprint

# Loading a yaml file
with open('params.yml') as f:
    d = yaml.safe_load(f)

pprint(d)


# Dumping a yaml file
data = {
    'mylist': [1, 42, 3.141, 1337, 'help', u'€'],
    'mystring': 'bla',
    'mynumber': 14,
    'mydict': {
        'foo': 'bar',
        'key': 'value',
        'the answer': 42
    }
}

with open('data-output.yaml', 'w', encoding='utf8') as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
