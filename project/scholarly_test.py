from scholarly import scholarly, ProxyGenerator



author = scholarly.search_pubs('Perception of physical stability and center of mass of 3D objects')
scholarly.pprint(next(author))