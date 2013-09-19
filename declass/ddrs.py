"""
Managing and formatting of DDRS document.
"""
import os

import declass.utils.database as db
import declass.utils.filefilter as ff


def make_db_connect():
    """
    Initialize a standard DB connection.
    """
    return db.DBCONNECT(host_name='mysql.csail.mit.edu', 
                        db_name='declassification',
                        user_name='declass', 
                        pwd='declass')


class Document(object):
    """
    Class to hold and process documents from the ddrs collection.
    """
    def __init__(self, id, unformatted_text):
        """
        Parameters
        ----------
        id : Int
             The document identifier.

        unformatted_text : String
             The unformatted (original) text of the document with markup.
        """
        self.id = id
        self.unformatted_text = unformatted_text
        self.clean_text = self._clean(self.unformatted_text)

        pages = self.clean_text.split("*PAGE*")
        self.sectioned_pages = []
        for page in pages:
            section_text = page.split("*FOOTER*")
            sections = {"body" : section_text[0]}
            if len(sections) == 2:
                sections["footer"] = section_text[1]
            else:
                sections["footer"] = ""
            self.sectioned_pages.append(sections)

    def _clean(self, text):
        """
        Remove the markup in the text.

        Parameters
        ----------
        text : String
            The unformatted text with markup.

        Returns
        -------
        String
            The cleaned text.
        """
        return text.replace("</PARA>", " \n ") \
            .replace("<PARA>", " \n ") \
            .replace("<?BR?>", " \n ") \
            .replace("<?PRE?>", " *PAGE* ") \
            .replace("<?HR?>", " *FOOTER* ") \
            .replace("</DOC.BODY>", " ") \
            .replace("<DOC.BODY>", " ") \
            .replace("\\n", " \n ")

    def format(self, output_spec):
        """
        Transform the text in a specific format.

        Parameters
        ----------
        output_spec : String
            The type of the formatting. Including:
            clean -> Remove all the formatting.
            nofoot -> Remove all the formatting and footers.
            raw -> Original text with markup.

        Returns
        -------
        String
            The formatted text.
        """
        if output_spec == "clean":
            return self.clean_text    
        elif output_spec == "nofoot":
            body_text = [page["body"] for page in self.sectioned_pages]
            return " *PAGE* ".join(body_text)
        elif output_spec == "raw":
            return self.unformatted_text
        else:
            raise Exception("output_spec " + output_spec + " not recognized.")
                
    @staticmethod
    def fetch_from_files(directory, ids=None):
        """
        Fetch documents from the filesytem. 

        Parameters
        ----------
        directory : String
            The file directory with the raw documents.

        ids : Iterator
            The id's of the documents to fetch, or None to get all.

        Returns
        -------
        Iterator
            Document objects for each id in ids. 
        """
        if ids is None:
            path_iter = ff.get_paths(
                directory, file_type="*.raw.txt", get_iter=True)
            for path in path_iter:
                head, tail = os.path.split(path)
                id = int(tail.split('.')[0])
                with open(path) as in_doc:
                    yield Document(id, in_doc.read())
                
        else:
            for id in ids:
                file_name = "{}.raw.txt".format(id)
                path = os.path.join(directory, file_name)
                with open(path) as in_doc:
                    yield Document(id, in_doc.read())

    @staticmethod
    def write_to_files(directory, documents, output_spec):
        """
        Write documents to the filesytem. 

        Parameters
        ----------
        directory : String
            The file directory for the documents.

        documents : Iterable over a set of ddrs.Documents
            The documents to write out.

        output_spec : String
            Format to write out the documents in.
        """
        for doc in documents:
            file_name = "{}.{}.txt".format(doc.id, output_spec)
            path = os.path.join(directory, file_name)
            with open(path, "w") as out:
                out.write(doc.format(output_spec))


if __name__ == "__main__":
    dbCon = make_db_connect()
    rows = dbCon.run_query("SELECT id, body FROM Document LIMIT 10")

    
    documents = (Document(row["id"], row["body"]) for row in rows)
    Document.write_to_files("/tmp/", documents, "raw")

    rows = dbCon.run_query("SELECT id FROM Document LIMIT 10")
    ids = (row["id"] for row in rows)
    file_docs = list(Document.fetch_from_files("/tmp/", ids))
    
    rows = dbCon.run_query("SELECT id, body FROM Document LIMIT 10")
    sql_docs = [Document(row["id"], row["body"]) for row in rows]

    assert len(sql_docs) == len(file_docs) 
    for file_doc, sql_doc in zip(file_docs, sql_docs):
        assert file_doc.id == sql_doc.id
        assert file_doc.format("raw") == sql_doc.format("raw")
