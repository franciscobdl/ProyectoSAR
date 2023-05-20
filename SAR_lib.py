import json
from nltk.stem.snowball import SnowballStemmer
import os
import re
import sys
import math
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle

class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """
    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [
        ("all", True), ("title", True), ("summary", True), ("section-name", True), ('url', False),
    ]
    def_field = 'all'
    PAR_MARK = '%'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10

    all_atribs = ['urls', 'index', 'sindex', 'ptindex', 'docs', 'weight', 'articles',
                  'tokenizer', 'stemmer', 'show_all', 'use_stemming', 'numindex']

    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes añadir más variables si las necesitas 

        """
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.numindex = {} # hash para el indice invertido de terminos pero con sus posiciones en los articulos --> clave: termino, valor: posting list (art_id, pos)
        self.sindex = {} # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {} # hash para el indice permuterm.
        self.docs = {} # diccionario de terminos --> clave: entero(docid), valor: ruta del fichero.
        self.weight = {} # hash de terminos para el pesado, ranking de resultados.
        self.articles = {} # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero (url, title, all)
        self.tokenizer = re.compile("\W+") # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer('spanish') # stemmer en castellano
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()


    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_snippet(self, v:bool):
        """

        Cambia el modo de mostrar snippet.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_snippet es True se mostrara un snippet de cada noticia, no aplicable a la opcion -C

        """
        self.show_snippet = v


    def set_stemming(self, v:bool):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v



    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):
        """
        Guarda la información del índice en un fichero en formato binario
        
        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """
        Carga la información del índice desde un fichero en formato binario
        
        """
        #info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls


    def index_dir(self, root:str, **args):
        """
        
        Recorre recursivamente el directorio o fichero "root" 
        NECESARIO PARA TODAS LAS VERSIONES
        
        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.multifield = args['multifield']
        self.positional = args['positional']
        self.stemming = args['stem']
        self.permuterm = args['permuterm']

        file_or_dir = Path(root)
        
        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in files:
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################
        
        
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos 
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article
                
    
    def index_file(self, filename:str):
        """

        Indexa el contenido de un fichero.
        
        input: "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.multifield y self.positional se debe ampliar el indexado


        """
        doc_id = len(self.docs) + 1
        self.docs[doc_id] = filename #da el docid
        
        for i, line in enumerate(open(filename)): #cada linea es un articulo. la i indica la pos del articulo en el archivo
            j = self.parse_article(line) #j = diccionario con la info del articulo
            if self.already_in_index(j):
                continue
            # print(j['all'])
            art_id = len(self.articles) + 1
            self.articles[art_id] = (j['url'], j['title'], j['all']) #asigna el art_id TODO: usa el valor de art_id para meter toda la info que necesites en forma de tupla

            # Indexar los campos
            # for field, tokenize_field in self.fields:
            #     if field not in j:
            #         continue
                
            content = j['all']
            tokens = enumerate(self.tokenize(content))
            # if tokenize_field else [content]
            
            for pos, token in tokens:
                term = token.lower() #pasa a minusculas

                if self.use_stemming:
                    term = self.stemmer.stem(term)

                if term not in self.index:
                    self.index[term] = [] #crea la entrada en el indice

                if term not in self.numindex:
                    self.numindex[term] = [] #crea la entrada en el index de posicion de terminos

                if art_id not in self.index[term]:
                    self.index[term].append(art_id)
                    
                self.numindex[term].append((art_id, pos)) #añade el art_id a la lista de art_id del termino con la posición en la que está

            self.urls.add(j['url'])

        
        if self.use_stemming:
            self.make_stemming()

        if self.permuterm:
            self.make_permuterm()
            
        terminos = sorted(self.index.keys())
        inverted_index = {}
        for termino in terminos:
            inverted_index[termino] = self.index[termino]
            
        self.index = inverted_index
        print(self.tokenize(self.articles[99][2])[1172])

        #
        # 
        # En la version basica solo se debe indexar el contenido "article"
        #
        #
        #
        #################
        ### COMPLETAR ###
        #################
        
    def get_token_pos(self, art_id, pos):
        #NUEVO
        """
        Obtiene el token dada una posición y un id de documento.
        
        input: "pos" es un entero que indica la posición del token en el documento "art_id"
        
        return: el token de la posición "pos" del documento "art_id"
        """
        return self.tokenize(self.articles[art_id][2])[pos]



    def set_stemming(self, v:bool):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v


    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()


    def make_stemming(self):
        """

        Crea el indice de stemming (self.sindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE STEMMING.

        "self.stemmer.stem(token) devuelve el stem del token"


        """
        terminos = list(self.index.keys())
        for termino in terminos:
            if self.stemmer.stem(termino) not in self.sindex: #si el stem no está en el indice de stemming
                self.sindex[self.stemmer.stem(termino)] = [] #crea la entrada en el indice de stemming
            else:
                self.sindex[self.stemmer.stem(termino)].append(termino)

        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################


    
    def make_permuterm(self):
        """

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE PERMUTERM


        """
        pass
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################




    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Muestra estadisticas de los indices
        
        """
        print("=" * 30)
        print("Number of indexed files:", len(self.docs))
        print("-" * 30)
        print("Number of indexed articles:", len(self.articles))
        print("-" * 30)
        print('TOKENS:')
        for field, tok in self.fields:
            if (self.multifield or field == "article"):
                print("\t# of tokens in '{}': {}".format(field, len(self.index[field])))
        if (self.permuterm):
            print("-" * 30)
            print('PERMUTERMS:')
            for field, tok in self.fields:
                if (self.multifield or field == "article"):
                    print("\t# of tokens in '{}': {}".format(field, len(self.ptindex[field])))
        if (self.stemming):
            print("-" * 30)
            print('STEMS:')
            for field, tok in self.fields:
                if (self.multifield or field == "article"):
                    print("\t# of tokens in '{}': {}".format(field, len(self.sindex[field])))
        print("-" * 30)
        if (self.positional):
            print('Positional queries are allowed.')
        else:    
            print('Positional queries are NOT allowed.')
        print("=" * 30)

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        



    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query:str, prev:Dict={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """
        query = self.tokenize(query)
       
        if query is None or len(query) == 0:
            return []
                

        operator = ['or', 'and', 'not']
        
        aux = []
        for w in query:
            if w not in operator:
                aux.append(w)
                if len(aux) > 1:
                    print('Error: hay dos palabras seguidas en la query')
                    return []
            else: aux = []
            
        print('query: ', query)

        # try:
        for i in range(len(query)):
            # print('query actual: ', query)
            if query[i] not in operator and len(query) == 1:
                query[i] = self.get_posting(query[i]) #si la query solo es una palabra, devuelve la posting list de esa palabra
                
            elif query[i] == 'or':
                if i - 1 == -1:
                    print('Error: no puede haber un operador al principio de la query')
                    return []
                elif query[i - 1] in operator or query[i + 1] == 'and':
                    print('Error: no puede haber dos operadores binarios seguidos')
                    return []
                elif query[i + 1] == 'not': #or not
                    query[i + 1] = self.reverse_posting(self.get_posting(query[i + 2]))
                    query[i + 2] = self.or_posting(self.get_posting(query[i - 1]), self.get_posting(query[i + 1]))
                    query[i - 1] = ''
                    query[i] = ''
                    query[i + 1] = ''
                else: 
                    query[i + 1] = self.or_posting(self.get_posting(query[i - 1]), self.get_posting(query[i + 1])) 
                    query[i - 1] = ''
                    query[i] = ''
                #el resultado lo guarda en la posicion mas a la derecha de los elementos implicados
            
            elif query[i] == 'and':
                if i - 1 == -1:
                    print('Error: no puede haber un operador al principio de la query')
                    return []
                elif query[i - 1] in operator or query[i + 1] == 'or':
                    print('Error: no puede haber dos operadores binarios seguidos')
                    return []
                elif query[i + 1] == 'not': #and not
                    query[i + 2] = self.minus_posting(self.get_posting(query[i - 1]), self.get_posting(query[i + 2]))
                    query[i]= '' #elimino el termino de la query para que no se vuelva a usar
                    query[i - 1]= '' #elimino el termino de la query para que no se vuelva a usar
                    query[i + 1]= '' #elimino el termino de la query para que no se vuelva a usar
                else: 
                    query[i + 1] = self.and_posting(self.get_posting(query[i - 1]), self.get_posting(query[i + 1]))
                    query[i]= '' #elimino el termino de la query para que no se vuelva a usar
                    query[i - 1]= '' #elimino el termino de la query para que no se vuelva a usar

            elif query[i] == 'not':
                if query[i + 1] in operator:
                    print('Error: no puede haber un operador despues de not')
                    return []            
                else: 
                    query[i + 1] = self.reverse_posting(self.get_posting(query[i + 1]))
                    query[i]= '' #elimino el termino de la query para que no se vuelva a usar

        return query[-1] #el resultado se queda en la ultima posicion de la lista
            
        # except IndexError:
        #     print('Error: la query está mal escrita')
        #     return []


    def get_posting(self, term:str, field:Optional[str]=None):
        """

        Devuelve la posting list asociada a un termino. 
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la amplaicion de stemming


        param:  "term": termino del que se debe recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario si se hace la ampliacion de multiples indices

        return: posting list
        
        NECESARIO PARA TODAS LAS VERSIONES

        """
        
        if isinstance(term, List): return term
        if term in self.index: return self.index[term]
        else: return []
        



    def get_positionals(self, terms:str, index):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        pass
        ########################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE POSICIONALES ##
        ########################################################


    def get_stemming(self, term:str, field: Optional[str]=None):
        """

        Devuelve la posting list asociada al stem de un termino.
        NECESARIO PARA LA AMPLIACION DE STEMMING

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        
        stem = self.stemmer.stem(term)

        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################

    def get_permuterm(self, term:str, field:Optional[str]=None):
        """

        Devuelve la posting list asociada a un termino utilizando el indice permuterm.
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """

        ##################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA PERMUTERM ##
        ##################################################
        pass



    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """
        
        #Conjunto de todos los Doc_IDs
        articulos = list(self.articles.keys())
        respuesta = []
        #Si la lista p contiene todos los Doc_IDs se devuelve una lista vacía
        #En principio, ninguna lista será mayor que la lista articulos
        if(p == articulos): return []
        
        #Si la lista es vacía, devuelve el conjunto entero
        if(len(p) == 0): return articulos

        for x in p:
            articulos.remove(x)
            
        return articulos


    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos en p1 y p2

        """
        #Inicialización de variables
        respuesta = []
        puntero1 = 0
        puntero2 = 0
        
        #Si alguna lista es vacía, devuelve una lista vacía
        if(len(p1) == 0 or len(p2) == 0): return respuesta
        
        #Bucle principal para recorrer las dos listas
        while(puntero1 < len(p1) and puntero2 < len(p2)):
            #Si los ID de los Documentos son iguales, se añade el documento a la respuesta y se avanzan los punteros
            if p1[puntero1] == p2[puntero2]:
                respuesta.append(p1[puntero1])
                puntero1 = puntero1 + 1
                puntero2 = puntero2 + 1
            else:
                #Si el ID del Documento de p1 es menor, se avanza el puntero de p1
                if p1[puntero1] < p2 [puntero2]:
                    puntero1 = puntero1 + 1
                #Sino, se avanza el de p2
                else:
                    puntero2 = puntero2 + 1
        #Se devuelve el resultado cuando cualquier puntero llega al final de la lista
        return respuesta
                
        


    def or_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el OR de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 o p2

        """
        #Inicialización de variables
        respuesta = []
        puntero1 = 0
        puntero2 = 0
        
        #Si alguna de las listas es vacía, devuelve la lista no vacía
        if(len(p1) == 0): return p2
        if(len(p2) == 0): return p1
        
        #Bucle principal para recorrer las dos listas
        while(puntero1 < len(p1) and puntero2 < len(p2)):
            #Si los ID de los Documentos son iguales, se añade el documento una sola vez a la respuesta y se avanzan los punteros
            if p1[puntero1] == p2[puntero2]:
                respuesta.append(p1)
                puntero1 = puntero1 + 1
                puntero2 = puntero2 + 1
            if p1[puntero1] < p2[puntero2]:
                respuesta.append(p1)
                puntero1 = puntero1 + 1
                while(p1[puntero1] < p2[puntero2]):
                    respuesta.append(p1[puntero1])
                    puntero1 = puntero1 + 1
                puntero1 = puntero1 + 1
                puntero2 = puntero2 + 1
            if p2[puntero2] < p1[puntero1]:
                respuesta.append(p2)
                puntero2 = puntero2 + 1
                while(p2[puntero2] < p1[puntero1]):
                    respuesta.append(p2[puntero2])
                    puntero2 = puntero2 + 1
                puntero1 = puntero1 + 1
                puntero2 = puntero2 + 1

        
        #Se añade la lista cuyo puntero no había llegado al final
        if(puntero1 < len(p1)):
            for docid in p1[puntero1:-1]:
                respuesta.append(docid)
        else:
            for docid in p2[puntero2:-1]:
                respuesta.append(docid)
                
        return respuesta


    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """
        #la resta es lo mismo que A AND NOT B
        p2 = self.reverse_posting(p2)
        return self.and_posting(p1, p2)

        
        pass
        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################


    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result = len(self.solve_query(query))
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True                    
            else:
                print(line)
        return not errors


    def solve_and_show(self, query:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados 

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        operator = ['or', 'and', 'not']
        pl = self.solve_query(query)
        n = len(pl)
        query = query.lower().split()
        count = 0
        for ord, art_id in enumerate(pl):
            print(f"Resultado {ord+1}\t({art_id})\tURL :{self. articles[art_id][0]}")
            print(f"\t{self.articles[art_id][1]}\n")
            if self.show_snippet:
                for q in query:
                    if q not in operator: #para cada palabra de la consulta que no sea un operador
                        (s1, s2) = self.get_snippet(art_id, q)
                        print(f"Snippet 1: ...{s1}...\n")
                        if s2 != '':
                            print(f"Snippet 2: ...{s2}...\n")
            count += 1
            if not self.show_all and count == self.SHOW_MAX: break #si no se quiere mostrar todos los resultados, se muestra hasta el maximo
            
        return n
        
        ################
        ## COMPLETAR  ##
        ################

    def get_snippet(self, art_id, term):
        #NUEVO
        """
        obtiene el snippet de un termino en un articulo 

        param:  art_id: id del articulo
                term: termino a buscar

        return: tupla con la primera y ultima aparicion del termino
        """
        posiciones = []
        articulos = self.numindex[term] #lista de tuplas (art_id, posiciones)
        for a in articulos:
            if a[0] == art_id:
                posiciones.append(a[1]) #si el articulo q busco coincide con el que estoy mirando, añado sus posiciones a la lista
        posi = posiciones[0]

        s1 = ''
        tokens = self.tokenize(self.articles[art_id][2])[posi-10:posi+10]
        for t in tokens:
            s1 += t + ' '
        s2 = '' 
        if len(posiciones) > 1:
            posf = posiciones[-1]
            tokens = self.tokenize(self.articles[art_id][2])[posf-10:posf+10]
            for t in tokens:
                s2 += t + ' '
        res = (s1, s2) #(snippet inicial,snippet final)
        

        return res








        

