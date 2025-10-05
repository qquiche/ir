package ir.vsr;

import java.io.File;
import java.util.*;
import ir.utilities.*;
import ir.classifiers.*;

/**
 * Proximity-enhanced inverted index that improves retrieval by considering
 * the closeness and order of query terms in documents.
 *
 * This index builds on a baseline inverted index with cosine similarity by
 * adding a positional component. The positional component measures how closely
 * the query terms appear to each other in each document and penalizes reversed
 * order. The final score divides the baseline cosine score by the average
 * proximity distance so that lower (better) proximity distances increase rank.</p>
 *
 */
public class InvertedPosIndex {

  /** Maximum number of retrievals to display per page. */
  public static final int MAX_RETRIEVALS = 10;

  /** Penalty multiplier applied when terms appear in the opposite order to the query. */
  private static final double ORDER_PENALTY = 2.0;

  /** Fallback distance assigned when a term pair cannot be matched in a document. */
  private static final double MAX_DISTANCE = 1000.0;

  /** Baseline index mapping tokens to their document-level statistics and postings. */
  public Map<String, TokenInfo> tokenHash = new HashMap<>();

  /** References to all indexed documents. */
  public List<DocumentReference> docRefs = new ArrayList<>();

  /** Directory containing the dataset to index. */
  public File dirFile;

  /** Document type selector (e.g., {@link DocumentIterator#TYPE_TEXT} or {@link DocumentIterator#TYPE_HTML}). */
  public short docType = DocumentIterator.TYPE_TEXT;

  /** Whether stemming is applied during tokenization. */
  public boolean stem = false;

  /** Whether interactive relevance feedback is enabled. */
  public boolean feedback = false;

  /**
   * A positional posting for a single term in a single document.
   * Holds the document reference, term frequency, and the sorted list of positions.
   */
  public static class PosPosting {
    /** Document containing the term. */
    final DocumentReference docRef;
    /** Term frequency in the document. */
    final int tf;
    /** Sorted positions of the term within the document after filtering. */
    final int[] positions;

    /**
     * Creates a positional posting.
     *
     * @param d document reference
     * @param tf term frequency in the document
     * @param p sorted positions where the term occurs
     */
    PosPosting(DocumentReference d, int tf, int[] p) {
      this.docRef = d;
      this.tf = tf;
      this.positions = p;
    }
  }

  /**
   * Positional token information stored in the positional index,
   * including IDF (mirrored from baseline index) and a list of positional postings.
   */
  public static class PosTokenInfo {
    /** Inverse document frequency of the token. */
    double idf = 0.0;
    /** List of positional postings for this token. */
    final List<PosPosting> occList = new ArrayList<>();
  }

  /** Positional index mapping tokens to their positional statistics. */
  public Map<String, PosTokenInfo> posTokenHash = new HashMap<>();

  /**
   * Constructs an inverted positional index from a directory of documents.
   *
   * @param dirFile dataset directory
   * @param docType document type flag (text or HTML)
   * @param stem whether to apply stemming
   * @param feedback whether to enable interactive relevance feedback
   * @throws IllegalStateException if indexing is invoked more than once
   */
  public InvertedPosIndex(File dirFile, short docType, boolean stem, boolean feedback) {
    this.dirFile = dirFile;
    this.docType = docType;
    this.stem = stem;
    this.feedback = feedback;
    indexDocuments();
  }

  /**
   * Constructs an inverted positional index from labeled examples.
   *
   * @param examples list of examples that provide documents and vectors
   * @throws IllegalStateException if indexing is invoked more than once
   */
  public InvertedPosIndex(List<Example> examples) {
    indexDocuments(examples);
  }

  /**
   * Builds the baseline and positional indexes from all documents in {@link #dirFile}.
   *
   * @throws IllegalStateException if indexes are already populated
   */
  protected void indexDocuments() {
    if (!tokenHash.isEmpty() || !docRefs.isEmpty()) {
      throw new IllegalStateException("Cannot indexDocuments more than once.");
    }

    DocumentIterator docIter = new DocumentIterator(dirFile, docType, stem);
    System.out.println("Indexing documents in " + dirFile);

    while (docIter.hasMoreDocuments()) {
      FileDocument doc = docIter.nextDocument();
      System.out.print(doc.file.getName() + ",");
      HashMapVector vector = doc.hashMapVector();
      indexDocument(doc, vector);
    }

    computeIDFandDocumentLengths();
    System.out.println("\nIndexed " + docRefs.size() + " documents with " + size() + " unique terms.");
  }

  /**
   * Builds the baseline and positional indexes from a set of examples.
   *
   * @param examples list of examples to index
   * @throws IllegalStateException if indexes are already populated
   */
  protected void indexDocuments(List<Example> examples) {
    if (!tokenHash.isEmpty() || !docRefs.isEmpty()) {
      throw new IllegalStateException("Cannot indexDocuments more than once.");
    }

    for (Example example : examples) {
      FileDocument doc = example.getDocument();
      HashMapVector vector = example.getHashMapVector();
      indexDocument(doc, vector);
    }

    computeIDFandDocumentLengths();
    System.out.println("Indexed " + docRefs.size() + " documents with " + size() + " unique terms.");
  }

  /**
   * Indexes a single document into the baseline inverted index and the positional index.
   *
   * @param doc file-backed document to index
   * @param vector term-frequency vector for the document
   */
  protected void indexDocument(FileDocument doc, HashMapVector vector) {
    DocumentReference docRef = new DocumentReference(doc);
    docRefs.add(docRef);

    for (Map.Entry<String, Weight> entry : vector.entrySet()) {
      String token = entry.getKey();
      int count = (int) entry.getValue().getValue();
      TokenInfo tokenInfo = tokenHash.get(token);
      if (tokenInfo == null) {
        tokenInfo = new TokenInfo();
        tokenHash.put(token, tokenInfo);
      }
      tokenInfo.occList.add(new TokenOccurrence(docRef, count));
    }

    Document posDoc = createRawDocument(doc.file);
    Map<String, List<Integer>> term2pos = new HashMap<>();
    int filteredPos = 0;

    while (posDoc.hasMoreTokens()) {
      String tok = posDoc.nextToken();
      if (tok == null) break;
      term2pos.computeIfAbsent(tok, k -> new ArrayList<>()).add(filteredPos++);
    }

    for (Map.Entry<String, List<Integer>> e : term2pos.entrySet()) {
      int[] pos = e.getValue().stream().mapToInt(Integer::intValue).toArray();
      Arrays.sort(pos);
      PosTokenInfo pti = posTokenHash.computeIfAbsent(e.getKey(), k -> new PosTokenInfo());
      pti.occList.add(new PosPosting(docRef, pos.length, pos));
    }
  }

  /**
   * Creates an appropriate raw document reader for positional indexing,
   * matching the tokenizer/stemming configuration of the baseline index.
   *
   * @param file input file
   * @return a raw {@link Document} for token iteration
   */
  private Document createRawDocument(File file) {
    if (docType == DocumentIterator.TYPE_HTML) {
      return new RawHTMLFileDocument(file, stem);
    } else {
      return new RawTextFileDocument(file, stem);
    }
  }

  /**
   * Computes IDF for all tokens, accumulates and normalizes document lengths for cosine similarity,
   * and mirrors IDF values into the positional index for corresponding tokens.
   */
  protected void computeIDFandDocumentLengths() {
    double N = docRefs.size();
    Iterator<Map.Entry<String, TokenInfo>> it = tokenHash.entrySet().iterator();

    while (it.hasNext()) {
      Map.Entry<String, TokenInfo> entry = it.next();
      TokenInfo tokenInfo = entry.getValue();
      double numDocRefs = tokenInfo.occList.size();
      double idf = Math.log(N / numDocRefs);

      if (idf == 0.0) {
        it.remove();
      } else {
        tokenInfo.idf = idf;
        for (TokenOccurrence occ : tokenInfo.occList) {
          occ.docRef.length += Math.pow(idf * occ.count, 2);
        }
      }
    }

    for (DocumentReference docRef : docRefs) {
      docRef.length = Math.sqrt(docRef.length);
    }

    for (Map.Entry<String, TokenInfo> e : tokenHash.entrySet()) {
      PosTokenInfo pti = posTokenHash.get(e.getKey());
      if (pti != null) {
        pti.idf = e.getValue().idf;
      }
    }
  }

  /**
   * Retrieves ranked documents for a raw string query, using both cosine
   * similarity and proximity enhancement.
   *
   * @param input query string
   * @return ranked retrieval results
   */
  public Retrieval[] retrieve(String input) {
    TextStringDocument orderDoc = new TextStringDocument(input, stem);
    List<String> qOrder = extractOrderedUniqueTokens(orderDoc);

    TextStringDocument vectorDoc = new TextStringDocument(input, stem);
    HashMapVector qv = vectorDoc.hashMapVector();

    return retrieve(qv, qOrder);
  }

  /**
   * Retrieves ranked documents for a query represented as a {@link Document}.
   *
   * @param doc query as a tokenized document
   * @return ranked retrieval results
   */
  public Retrieval[] retrieve(Document doc) {
    return retrieve(doc.hashMapVector());
  }

  /**
   * Retrieves ranked documents for a query represented as a {@link HashMapVector}.
   * The term order is derived from the vector's key set (sorted).
   *
   * @param vector query vector
   * @return ranked retrieval results
   */
  public Retrieval[] retrieve(HashMapVector vector) {
    List<String> qTerms = new ArrayList<>(vector.hashMap.keySet());
    Collections.sort(qTerms);
    return retrieve(vector, qTerms);
  }

  /**
   * Core retrieval method combining baseline cosine similarity with a proximity score.
   * The final score is computed as {@code cosine / proximityDistance}, where a lower
   * average proximity distance (better proximity) increases rank.
   *
   * @param queryVector query term-frequency vector
   * @param qOrder ordered list of unique query terms as they appeared in the query
   * @return ranked retrieval results
   */
  protected Retrieval[] retrieve(HashMapVector queryVector, List<String> qOrder) {
    Map<DocumentReference, DoubleValue> retrievalHash = new HashMap<>();
    double queryLength = 0.0;

    for (Map.Entry<String, Weight> entry : queryVector.entrySet()) {
      String token = entry.getKey();
      double count = entry.getValue().getValue();
      queryLength += incorporateToken(token, count, retrievalHash);
    }
    queryLength = Math.sqrt(queryLength);

    RetrievalProx[] candidates = new RetrievalProx[retrievalHash.size()];
    int idx = 0;

    for (Map.Entry<DocumentReference, DoubleValue> e : retrievalHash.entrySet()) {
      DocumentReference d = e.getKey();
      double cosine = e.getValue().value / (queryLength * d.length);
      candidates[idx++] = new RetrievalProx(d, cosine, cosine, 0.0);
    }

    for (RetrievalProx rp : candidates) {
      double proximityScore = computeProximityScore(qOrder, rp.docRef);
      rp.prox = proximityScore;
      rp.score = rp.cosine / proximityScore;
    }

    Arrays.sort(candidates);
    return candidates;
  }

  /**
   * Computes the average closest distance among all unordered pairs of unique query terms,
   * measured across their occurrences within a document. If the dominant local occurrence
   * order contradicts the query term order, the distance is multiplied by {@link #ORDER_PENALTY}.
   *
   * Returns 1.0 for single-term or single-unique-term queries.
   *
   * @param qOrder ordered list of unique query terms as they appeared in the query
   * @param docRef document to evaluate
   * @return average pairwise closest distance (lower is better)
   */
  protected double computeProximityScore(List<String> qOrder, DocumentReference docRef) {
    if (qOrder.size() < 2) {
      return 1.0;
    }

    List<String> uniqueTerms = new ArrayList<>();
    Set<String> seen = new HashSet<>();
    for (String term : qOrder) {
      if (seen.add(term)) {
        uniqueTerms.add(term);
      }
    }

    if (uniqueTerms.size() < 2) {
      return 1.0;
    }

    List<int[]> posLists = new ArrayList<>();
    for (String term : uniqueTerms) {
      int[] positions = getPositions(term, docRef);
      posLists.add(positions != null ? positions : new int[0]);
    }

    double totalDistance = 0.0;
    int pairCount = 0;

    for (int i = 0; i < uniqueTerms.size(); i++) {
      for (int j = i + 1; j < uniqueTerms.size(); j++) {
        int[] pos1 = posLists.get(i);
        int[] pos2 = posLists.get(j);
        boolean expectForwardOrder = qOrder.indexOf(uniqueTerms.get(i)) < qOrder.indexOf(uniqueTerms.get(j));
        double pairDistance = computeClosestPairDistance(pos1, pos2, expectForwardOrder);
        totalDistance += pairDistance;
        pairCount++;
      }
    }

    double avgDistance = pairCount > 0 ? totalDistance / pairCount : MAX_DISTANCE;
    return avgDistance;
  }

  /**
   * Computes the closest distance between occurrences of two terms within a document,
   * applying an order penalty when the nearest partner occurrence contradicts the
   * expected query order.
   *
   * @param pos1 sorted positions of term 1
   * @param pos2 sorted positions of term 2
   * @param expectForwardOrder true if term 1 is expected to appear before term 2 in the query
   * @return the smallest adjusted distance between any occurrence pair, or {@link #MAX_DISTANCE}
   *         if one of the term arrays is empty
   */
  protected double computeClosestPairDistance(int[] pos1, int[] pos2, boolean expectForwardOrder) {
    if (pos1.length == 0 || pos2.length == 0) {
      return MAX_DISTANCE;
    }

    double minDistance = MAX_DISTANCE;

    for (int p1 : pos1) {
      int insertionPoint = Arrays.binarySearch(pos2, p1);

      if (insertionPoint >= 0) {
        minDistance = Math.min(minDistance, 0.0);
      } else {
        insertionPoint = -insertionPoint - 1;

        if (insertionPoint < pos2.length) {
          int p2 = pos2[insertionPoint];
          double distance = Math.abs(p2 - p1);
          if (expectForwardOrder && p2 < p1) {
            distance *= ORDER_PENALTY;
          } else if (!expectForwardOrder && p2 > p1) {
            distance *= ORDER_PENALTY;
          }
          minDistance = Math.min(minDistance, distance);
        }

        if (insertionPoint > 0) {
          int p2 = pos2[insertionPoint - 1];
          double distance = Math.abs(p2 - p1);
          if (expectForwardOrder && p2 < p1) {
            distance *= ORDER_PENALTY;
          } else if (!expectForwardOrder && p2 > p1) {
            distance *= ORDER_PENALTY;
          }
          minDistance = Math.min(minDistance, distance);
        }
      }
    }

    return minDistance;
  }

  /**
   * Retrieves the array of positions for {@code term} within {@code docRef}.
   *
   * @param term token to look up
   * @param docRef target document reference
   * @return sorted positions array, or {@code null} if the term does not occur
   */
  protected int[] getPositions(String term, DocumentReference docRef) {
    PosTokenInfo pti = posTokenHash.get(term);
    if (pti == null) return null;

    for (PosPosting posting : pti.occList) {
      if (posting.docRef.equals(docRef)) {
        return posting.positions;
      }
    }
    return null;
  }

  /**
   * Incorporates a single query token into the retrieval accumulator,
   * updating per-document dot products and returning the squared
   * query weight contribution to the query length.
   *
   * @param token query token
   * @param count token count in the query
   * @param retrievalHash accumulator mapping documents to partial dot products
   * @return squared query weight contribution ({@code (idf*count)^2}) or 0 if token unseen
   */
  protected double incorporateToken(String token, double count,
                                    Map<DocumentReference, DoubleValue> retrievalHash) {
    TokenInfo tokenInfo = tokenHash.get(token);
    if (tokenInfo == null) return 0.0;

    double weight = tokenInfo.idf * count;
    for (TokenOccurrence occ : tokenInfo.occList) {
      DoubleValue val = retrievalHash.get(occ.docRef);
      if (val == null) {
        val = new DoubleValue(0.0);
        retrievalHash.put(occ.docRef, val);
      }
      val.value += weight * tokenInfo.idf * occ.count;
    }
    return weight * weight;
  }

  /**
   * Extracts the list of unique tokens from a document in the order they first appear.
   *
   * @param d tokenized document
   * @return ordered list of unique tokens
   */
  protected static List<String> extractOrderedUniqueTokens(Document d) {
    List<String> order = new ArrayList<>();
    Set<String> seen = new HashSet<>();

    while (d.hasMoreTokens()) {
      String tok = d.nextToken();
      if (tok == null) break;
      if (seen.add(tok)) {
        order.add(tok);
      }
    }
    return order;
  }

  /**
   * Retrieval result with additional proximity diagnostics.
   * Stores the baseline cosine and the computed proximity distance separately.
   */
  public static class RetrievalProx extends Retrieval {
    /** Baseline cosine similarity score. */
    final double cosine;
    /** Average proximity distance (lower is better). */
    double prox;

    /**
     * Creates a proximity-augmented retrieval result.
     *
     * @param d document reference
     * @param finalScore final combined score (cosine/proximity)
     * @param cosine baseline cosine score
     * @param prox proximity distance
     */
    RetrievalProx(DocumentReference d, double finalScore, double cosine, double prox) {
      super(d, finalScore);
      this.cosine = cosine;
      this.prox = prox;
    }
  }

  /**
   * Runs an interactive query loop on standard input, supporting paging,
   * document viewing, and optional relevance feedback.
   */
  public void processQueries() {
    System.out.println("Now able to process queries. When done, enter an empty query to exit.");

    while (true) {
      String query = UserInput.prompt("\nEnter query:  ");
      if (query.equals("")) break;

      Retrieval[] retrievals = retrieve(query);
      presentRetrievals(new TextStringDocument(query, stem).hashMapVector(), retrievals);
    }
  }

  /**
   * Presents retrievals, supports paging, opening documents in a browser,
   * and applying relevance feedback when enabled.
   *
   * @param queryVector query vector used (possibly updated after feedback)
   * @param retrievals retrieval results to present
   */
  public void presentRetrievals(HashMapVector queryVector, Retrieval[] retrievals) {
    if (showRetrievals(retrievals)) {
      FeedbackProx fdback = null;
      if (feedback) fdback = new FeedbackProx(queryVector, retrievals, this);

      int currentPosition = MAX_RETRIEVALS;

      while (true) {
        String command = UserInput.prompt("\n Enter command:  ");
        if (command.equals("")) break;

        if (command.equals("m")) {
          printRetrievals(retrievals, currentPosition);
          currentPosition += MAX_RETRIEVALS;
          continue;
        }

        if (command.equals("r") && feedback) {
          if (fdback.isEmpty()) {
            System.out.println("Need to first view some documents and provide feedback.");
            continue;
          }
          System.out.println("Positive docs: " + fdback.goodDocRefs +
              "\nNegative docs: " + fdback.badDocRefs);
          System.out.println("Executing New Expanded and Reweighted Query: ");
          queryVector = fdback.newQuery();
          retrievals = retrieve(queryVector);
          fdback.retrievals = retrievals;
          if (showRetrievals(retrievals)) continue;
          else break;
        }

        try {
          int showNumber = Integer.parseInt(command);
          if (showNumber > 0 && showNumber <= retrievals.length) {
            System.out.println("Showing document " + showNumber + " in the " + Browser.BROWSER_NAME + " window.");
            Browser.display(retrievals[showNumber - 1].docRef.file);
            if (feedback && !fdback.haveFeedback(showNumber)) {
              fdback.getFeedback(showNumber);
            }
          } else {
            System.out.println("No such document number: " + showNumber);
          }
        } catch (NumberFormatException e) {
          System.out.println("Unknown command.");
          System.out.println("Enter `m' to see more, a number to show the nth document, nothing to exit.");
          if (feedback && !fdback.isEmpty()) {
            System.out.println("Enter `r' to use any feedback given to `redo' with a revised query.");
          }
        }
      }
    }
  }

  /**
   * Prints the header and the first page of retrievals, and displays
   * usage hints for paging and feedback.
   *
   * @param retrievals retrieval results
   * @return {@code true} if there are results; {@code false} otherwise
   */
  public boolean showRetrievals(Retrieval[] retrievals) {
    if (retrievals.length == 0) {
      System.out.println("\nNo matching documents found.");
      return false;
    } else {
      System.out.println("\nTop " + MAX_RETRIEVALS + " matching Documents from most to least relevant:");
      printRetrievals(retrievals, 0);
      System.out.println("\nEnter `m' to see more, a number to show the nth document, nothing to exit.");
      if (feedback) {
        System.out.println("Enter `r' to use any relevance feedback given to `redo' with a revised query.");
      }
      return true;
    }
  }

  /**
   * Prints a page of retrieval results starting from {@code start}, up to {@link #MAX_RETRIEVALS}.
   * If the retrieval is a {@link RetrievalProx}, it also prints the cosine and proximity components.
   *
   * @param retrievals retrieval results
   * @param start starting index (0-based)
   */
  public void printRetrievals(Retrieval[] retrievals, int start) {
    System.out.println("");
    if (start >= retrievals.length) {
      System.out.println("No more retrievals.");
      return;
    }

    for (int i = start; i < Math.min(retrievals.length, start + MAX_RETRIEVALS); i++) {
      Retrieval r = retrievals[i];
      if (r instanceof RetrievalProx) {
        RetrievalProx rp = (RetrievalProx) r;
        System.out.println(
            MoreString.padTo((i + 1) + ". ", 4) +
            MoreString.padTo(rp.docRef.file.getName(), 20) +
            " Score: " + MoreMath.roundTo(rp.score, 5) +
            " (Vector: " + MoreMath.roundTo(rp.cosine, 5) +
            "; Proximity: " + MoreMath.roundTo(rp.prox, 5) + ")"
        );
      } else {
        System.out.println(
            MoreString.padTo((i + 1) + ". ", 4) +
            MoreString.padTo(r.docRef.file.getName(), 20) +
            " Score: " + MoreMath.roundTo(r.score, 5)
        );
      }
    }
  }

  /**
   * Returns the number of unique terms in the baseline index.
   *
   * @return vocabulary size
   */
  public int size() { return tokenHash.size(); }

  /**
   * Clears all index structures and document references.
   */
  public void clear() {
    docRefs.clear();
    tokenHash.clear();
    posTokenHash.clear();
  }

  /**
   * Entry point. Accepts flags:
   * 
   *   {@code -html}: index HTML documents
   *   {@code -stem}: enable stemming
   *   {@code -feedback}: enable interactive relevance feedback
   *
   * The final argument must be the directory path to index.
   *
   * @param args command-line arguments
   * @throws IllegalArgumentException if an unknown flag is provided
   */
  public static void main(String[] args) {
    String dirName = args[args.length - 1];
    short docType = DocumentIterator.TYPE_TEXT;
    boolean stem = false, feedback = false;

    for (int i = 0; i < args.length - 1; i++) {
      String flag = args[i];
      if (flag.equals("-html")) {
        docType = DocumentIterator.TYPE_HTML;
      } else if (flag.equals("-stem")) {
        stem = true;
      } else if (flag.equals("-feedback")) {
        feedback = true;
      } else {
        throw new IllegalArgumentException("Unknown flag: " + flag);
      }
    }

    InvertedPosIndex index = new InvertedPosIndex(new File(dirName), docType, stem, feedback);
    index.processQueries();
  }
}
