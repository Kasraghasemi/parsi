from unittest import result
import parsi

a = { 'q':2,'w':3 ,'e':'t'}
b = { 'q':2,'w':3 ,'e':'t'}
result = parsi.concate_dictionaries([a,b])

assert result == {'q': [2, 2], 'w': [3, 3], 'e': ['t', 't']}