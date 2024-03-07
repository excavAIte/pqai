import itertools
from core.indexes import VectorIndex
from core.results import SearchResult

class Searcher():

    _invalid_needle_msg = "Invalid needle"
    _invalid_haystack_msg = "Invalid haystack"
    _no_haystack_msg = "No haystacks available for search"

    def __init__(self):
        self._compat_haystack_type = None

    def _needle_compatibility_fn(self, needle):
        raise NotImplementedError
    
    def _search_fn(self, needle, haystack, n):
        raise NotImplementedError
    
    def _sort_fn(self, results):
        return results

    def search(self, needle, haystack, n_results=10):
        self._check_needle_compatibility(needle)
        self._check_haystack_compatibility(haystack)
        haystack = [haystack] if self._is_one_haystack(haystack) else haystack
        n_results = max(0, n_results)
        return self._search_many(needle, haystack, n_results)

    def _check_needle_compatibility(self, needle):
        if not self._needle_compatibility_fn(needle):
            raise Exception(self._invalid_needle_msg)

    def _check_haystack_compatibility(self, haystack):
        if isinstance(haystack, self._compat_haystack_type):
            return None
        elif isinstance(haystack, list) or isinstance(haystack, set):
            if all(isinstance(item, self._compat_haystack_type) for item in haystack):
                return None
        else:
            raise Exception(self._invalid_haystack_msg)

    def _is_one_haystack(self, haystack):
        if isinstance(haystack, self._compat_haystack_type):
            return True
        if hasattr(haystack, "__iter__"):
            return False
        raise ValueError
    
    def _search_many(self, needle, haystack, n):
        list_of_lists = [self._search_one(needle, hs, n) for hs in haystack]
        results = self._flatten(list_of_lists)
        results = self._sort_fn(results)
        results = self._deduplicate(results)
        return results[:n]

    def _search_one(self, needle, haystack, n):
        return self._search_fn(needle, haystack, n)

    def _flatten(self, list2d):
        return list(itertools.chain.from_iterable(list2d))

    def _deduplicate(self, results):
        output = []
        added = set()
        for r in results:
            if len(output) > 0:
                previous = output[-1]
                if r.score == previous.score:
                    continue
                if r.id in added:
                    continue

            output.append(r)
            added.add(r.id)
        return output


class VectorIndexSearcher(Searcher):

    _invalid_needle_msg = "Expected a vector as query"
    _invalid_haystack_msg = "Can only search VectorIndex objects"
    _no_haystack_msg = "No VectorIndex objects provided for search"

    def __init__(self):
        super().__init__()
        self._compat_haystack_type = VectorIndex

    def _needle_compatibility_fn(self, needle):
        if not hasattr(needle, "__iter__"):
            for i in needle:
                try:
                    float(i)
                except ValueError:
                    return False
        return True

    def _search_fn(self, vector, index, n):
        pairs = index.search(vector, n)
        pairs = [(res_id, dist) for res_id, dist in pairs if 0.0 <= dist <= 2.0]
        # sort
        pairs = sorted(pairs, key=lambda x: x[1])
        index_id = index.name
        triplets = [(res_id, index_id, dist) for res_id, dist in pairs]
        results = [SearchResult(*triplet) for triplet in triplets]
        return results

    def _sort_fn(self, results):
        return sorted(results, key=lambda x: x.score)
