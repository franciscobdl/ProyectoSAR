#! -*- encoding: utf8 -*-
import heapq as hq

from typing import Tuple, List, Optional, Dict, Union

import requests
import bs4
import re
from urllib.parse import urljoin
import json
import math
import os

class SAR_Wiki_Crawler:

    def __init__(self):
        # Expresión regular para detectar si es un enlace de la Wikipedia
        self.wiki_re = re.compile(r"(http(s)?:\/\/(es)\.wikipedia\.org)?\/wiki\/[\w\/_\(\)\%]+")
        # Expresión regular para limpiar anclas de editar
        self.edit_re = re.compile(r"\[(editar)\]")
        # Formato para cada nivel de sección
        self.section_format = {
            "h1": "##{}##",
            "h2": "=={}==",
            "h3": "--{}--"
        }

        # Expresiones regulares útiles para el parseo del documento
        self.title_sum_re = re.compile(r"##(?P<title>.+)##\n(?P<summary>((?!==.+==).+|\n)+)(?P<rest>(.+|\n)*)")
        self.sections_re = re.compile(r"==.+==\n")
        #r"==(?P<name>.+)==\n(?P<text>((?!--.+--).+|\n)*)(?P<rest>(.+|\n)*)"
        self.section_re = re.compile(r"(?P<text>((?!--.+--).+|\n)*)(?P<rest>(.+|\n)*)") #texto es todo lo q haya antes de una subseccion
        self.subsections_re = re.compile(r"--.+--\n")
        self.subsection_re = re.compile(r"--(?P<name>.+)--\n(?P<text>(.+|\n)*)")


    def is_valid_url(self, url: str) -> bool:
        """Verifica si es una dirección válida para indexar

        Args:
            url (str): Dirección a verificar

        Returns:
            bool: True si es valida, en caso contrario False
        """
        return self.wiki_re.fullmatch(url) is not None


    def get_wikipedia_entry_content(self, url: str) -> Optional[Tuple[str, List[str]]]:
        """Devuelve el texto en crudo y los enlaces de un artículo de la wikipedia

        Args:
            url (str): Enlace a un artículo de la Wikipedia

        Returns:
            Optional[Tuple[str, List[str]]]: Si es un enlace correcto a un artículo
                de la Wikipedia en inglés o castellano, devolverá el texto y los
                enlaces que contiene la página.

        Raises:
            ValueError: En caso de que no sea un enlace a un artículo de la Wikipedia
                en inglés o español
        """
        if not self.is_valid_url(url):
            raise ValueError((
                f"El enlace '{url}' no es un artículo de la Wikipedia en español"
            ))

        try:
            req = requests.get(url)
        except Exception as ex:
            print(f"ERROR: - {url} - {ex}")
            return None


        # Solo devolvemos el resultado si la petición ha sido correcta
        if req.status_code == 200:
            soup = bs4.BeautifulSoup(req.text, "lxml")
            urls = set()

            for ele in soup.select((
                'div#catlinks, div.printfooter, div.mw-authority-control'
            )):
                ele.decompose()

            # Recogemos todos los enlaces del contenido del artículo
            for a in soup.select("div#bodyContent a", href=True):
                href = a.get("href")
                if href is not None:
                    urls.add(href)

            # Contenido del artículo
            content = soup.select((
                "h1.firstHeading,"
                "div#mw-content-text h2,"
                "div#mw-content-text h3,"
                "div#mw-content-text h4,"
                "div#mw-content-text p,"
                "div#mw-content-text ul,"
                "div#mw-content-text li,"
                "div#mw-content-text span"
            ))

            dedup_content = []
            seen = set()

            for element in content:
                if element in seen:
                    continue

                dedup_content.append(element)

                # Añadimos a vistos, tanto el elemento como sus descendientes
                for desc in element.descendants:
                    seen.add(desc)

                seen.add(element)

            text = "\n".join(
                self.section_format.get(element.name, "{}").format(element.text)
                for element in dedup_content
            )

            # Eliminamos el texto de las anclas de editar
            text = self.edit_re.sub('', text)

            return text, sorted(list(urls))

        return None


    def parse_wikipedia_textual_content(self, text: str, url: str) -> Optional[Dict[str, Union[str,List]]]:

        ##COMPLETAR
        """Devuelve una estructura tipo artículo a partir del text en crudo

        Args:
            text (str): Texto en crudo del artículo de la Wikipedia
            url (str): url del artículo, para añadirlo como un campo

        Returns:

            Optional[Dict[str, Union[str,List[Dict[str,Union[str,List[str,str]]]]]]]:

            devuelve un diccionario con las claves 'url', 'title', 'summary', 'sections':
                Los valores asociados a 'url', 'title' y 'summary' son cadenas,
                el valor asociado a 'sections' es una lista de posibles secciones.
                    Cada sección es un diccionario con 'name', 'text' y 'subsections',
                        los valores asociados a 'name' y 'text' son cadenas y,
                        el valor asociado a 'subsections' es una lista de posibles subsecciones
                        en forma de diccionario con 'name' y 'text'.

            en caso de no encontrar título o resúmen del artículo, devolverá None

        """
        def clean_text(txt): 
            return '\n'.join(l for l in txt.split('\n') if len(l) > 0) # Eliminamos líneas vacías

        title = self.title_sum_re.search(text)
        document = {}
        document['url'] = url
        document['title'] = title.group('title')
        document['summary'] = title.group('summary')
        document['sections'] = []

        section = self.sections_re.findall(text) #lista de secciones
        nsection = []
        #obtenemos la lista de las secciones limpia y lista para meterlas en el diccionario
        for s in section:
            s = s.strip().strip('==')
            nsection = nsection + [s]
       
        secciones = self.sections_re.split(text) #separamos el texto entero por secciones
        for i, seccion in enumerate(nsection):
            document['sections'].append({})
            document['sections'][i]['name'] = seccion
            aux = self.section_re.search(secciones[i+1]) #en el texto de las secciones solo debe figurar el suyo, sin subsecciones
            document['sections'][i]['text'] = clean_text(aux.group('text')) 
            document['sections'][i]['subsections'] = []
            subsection = self.subsections_re.findall(secciones[i+1]) #le pasamos el texto de la seccion con las subsecciones
            if len(subsection) > 0: #si no tiene subsecciones una seccion, su len es 0
                nsubsection = []
                #hacemos lo mismo que con las secciones, limpiamos los nombres
                for s in subsection:
                    s = s.strip().strip('--')
                    nsubsection = nsubsection + [s]
                subsecciones = self.subsections_re.split(secciones[i+1])
                
                for j, subseccion in enumerate(nsubsection):
                    document['sections'][i]['subsections'].append({})
                    document['sections'][i]['subsections'][j]['name'] = subseccion
                    document['sections'][i]['subsections'][j]['text'] = clean_text(subsecciones[j+1])
        print(document["sections"][2])

        #VA PERFECTO

        return document


    def save_documents(self,
        documents: List[dict], base_filename: str,
        num_file: Optional[int] = None, total_files: Optional[int] = None
    ):
        """Guarda una lista de documentos (text, url) en un fichero tipo json lines
        (.json). El nombre del fichero se autogenera en base al base_filename,
        el num_file y total_files. Si num_file o total_files es None, entonces el
        fichero de salida es el base_filename.

        Args:
            documents (List[dict]): Lista de documentos.
            base_filename (str): Nombre base del fichero de guardado.
            num_file (Optional[int], optional):
                Posición numérica del fichero a escribir. (None por defecto)
            total_files (Optional[int], optional):
                Cantidad de ficheros que se espera escribir. (None por defecto)
        """
        assert base_filename.endswith(".json")

        if num_file is not None and total_files is not None:
            # Separamos el nombre del fichero y la extensión
            base, ext = os.path.splitext(base_filename)
            # Padding que vamos a tener en los números
            padding = len(str(total_files))

            out_filename = f"{base}_{num_file:0{padding}d}_{total_files}{ext}"

        else:
            out_filename = base_filename

        with open(out_filename, "w", encoding="utf-8", newline="\n") as ofile:
            for doc in documents:
                print(json.dumps(doc, ensure_ascii=True), file=ofile)


    def start_crawling(self, 
                    initial_urls: List[str], document_limit: int,
                    base_filename: str, batch_size: Optional[int], max_depth_level: int,
                    ):     

        ##COMPLETAR   
         

        """Comienza la captura de entradas de la Wikipedia a partir de una lista de urls válidas, 
            termina cuando no hay urls en la cola o llega al máximo de documentos a capturar.
        
        Args:
            initial_urls: Direcciones a artículos de la Wikipedia
            document_limit (int): Máximo número de documentos a capturar
            base_filename (str): Nombre base del fichero de guardado.
            batch_size (Optional[int]): Cada cuantos documentos se guardan en
                fichero. Si se asigna None, se guardará al finalizar la captura.
            max_depth_level (int): Profundidad máxima de captura.
        """

        # URLs válidas, ya visitadas (se hayan procesado, o no, correctamente)
        visited = set()
        # URLs en cola
        to_process = set(initial_urls)
        # Direcciones a visitar
        queue = [(0, "", url) for url in to_process]
        hq.heapify(queue) # Ordenamos la cola por profundidad
        # Buffer de documentos capturados
        documents: List[dict] = [] #aqui se guardan los documentos q han sido procesados


        # Contador del número de documentos capturados
        total_documents_captured = 0
        # Contador del número de ficheros escritos
        files_count = 0

        # En caso de que no utilicemos bach_size, asignamos None a total_files
        # así el guardado no modificará el nombre del fichero base
        if batch_size is None:
            total_files = None
        else:
            # Suponemos que vamos a poder alcanzar el límite para la nomenclatura
            # de guardado
            total_files = math.ceil(document_limit / batch_size)

        for i in range(1): #TODO cambiar el range
            url = hq.heappop(queue) # Sacamos la primera url de la cola
            visited.add(url) # La añadimos a visitadas
            content, urls = self.get_wikipedia_entry_content('https://es.wikipedia.org/wiki/Videojuego') # Obtenemos el texto y las urls q tiene  #(url)

            #TODO: url puede q no sea de la wikipedia y lanzará una excepcion, hay q tratarla??
            documents.append(self.parse_wikipedia_textual_content(content, url)) #procesamos el contenido y lo añadimos al buffer
            total_documents_captured += 1 # Aumentamos el contador de documentos capturados

            
            for u in urls: # Añadimos las urls obtenidas q no han sido visitadas a la cola 
                if u not in visited:
                    hq.heappush(queue, u)

            if total_documents_captured >= document_limit: # Si hemos alcanzado el límite de documentos capturados, deja de hacer nada
                break
            

        #guarda en un fichero el documento de una url
        #crea una lista aux para ir metiendo batch_size documentos. Cuando se llegue a ese tamaño,
        #se manda ese aux a save_documents y se vacía. Si no llega al tamaño pero es el ultimo elemento, 
        #lo manda tmb q es el ultimo fragmento

        aux = []
        for i in documents:
            aux.append(i)
            if len(aux) == batch_size:
                self.save_documents(aux, base_filename, files_count, total_files) # Guardamos los documentos
                files_count += 1
                aux = []
            elif len(aux) < batch_size and i == documents[-1]:
                self.save_documents(aux, base_filename, files_count, total_files)
                


    def wikipedia_crawling_from_url(self,
        initial_url: str, document_limit: int, base_filename: str,
        batch_size: Optional[int], max_depth_level: int
    ):
        """Captura un conjunto de entradas de la Wikipedia, hasta terminar
        o llegar al máximo de documentos a capturar.
        
        Args:
            initial_url (str): Dirección a un artículo de la Wikipedia
            document_limit (int): Máximo número de documentos a capturar
            base_filename (str): Nombre base del fichero de guardado.
            batch_size (Optional[int]): Cada cuantos documentos se guardan en
                fichero. Si se asigna None, se guardará al finalizar la captura.
            max_depth_level (int): Profundidad máxima de captura.
        """
        if not self.is_valid_url(initial_url) and not initial_url.startswith("/wiki/"):
            raise ValueError(
                "Es necesario partir de un artículo de la Wikipedia en español"
            )

        self.start_crawling(initial_urls=[initial_url], document_limit=document_limit, base_filename=base_filename,
                            batch_size=batch_size, max_depth_level=max_depth_level)



    def wikipedia_crawling_from_url_list(self,
        urls_filename: str, document_limit: int, base_filename: str,
        batch_size: Optional[int]
    ):
        """A partir de un fichero de direcciones, captura todas aquellas que sean
        artículos de la Wikipedia válidos

        Args:
            urls_filename (str): Lista de direcciones
            document_limit (int): Límite máximo de documentos a capturar
            base_filename (str): Nombre base del fichero de guardado.
            batch_size (Optional[int]): Cada cuantos documentos se guardan en
                fichero. Si se asigna None, se guardará al finalizar la captura.

        """

        urls = []
        with open(urls_filename, "r", encoding="utf-8") as ifile:
            for url in ifile:
                url = url.strip()

                # Comprobamos si es una dirección a un artículo de la Wikipedia
                if self.is_valid_url(url):
                    if not url.startswith("http"):
                        raise ValueError(
                            "El fichero debe contener URLs absolutas"
                        )

                    urls.append(url)

        urls = list(set(urls)) # eliminamos posibles duplicados

        self.start_crawling(initial_urls=urls, document_limit=document_limit, base_filename=base_filename,
                            batch_size=batch_size, max_depth_level=0)





if __name__ == "__main__":
    raise Exception(
        "Esto es una librería y no se puede usar como fichero ejecutable"
    )
