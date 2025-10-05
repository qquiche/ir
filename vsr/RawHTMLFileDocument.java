package ir.vsr;

import java.io.*;

/**
 * HTML document reader for positional indexing.
 *
 * This class behaves like {@link HTMLFileDocument} with HTML tag stripping and optional
 * stemming, but intentionally does not remove stopwords. Retaining stopwords
 * preserves true token-to-token distances for accurate positional/proximity calculations.
 */
public class RawHTMLFileDocument extends HTMLFileDocument {

  /**
   * Creates a raw HTML document reader.
   *
   * @param file the HTML file to read
   * @param stem whether stemming should be applied to produced tokens
   */
  public RawHTMLFileDocument(File file, boolean stem) {
    super(file, stem);
  }

  /**
   * Advances {@code nextToken} to the next valid token.
   *
   * Processing steps:
   *
   *   Obtain the next candidate token from the underlying HTML stream
   *       (tags already stripped by the parent class).
   *   Normalize to lowercase.
   *   Filter non-letter tokens using {@code allLetters}.
   *   Optionally apply stemming via {@code stemmer.stripAffixes} and revalidate.
   *   Emit the token without stopword filtering.
   *
   * If no further tokens exist, {@code nextToken} is left as {@code null}.
   */
  @Override
  protected void prepareNextToken() {
    do {
      nextToken = getNextCandidateToken();
      if (nextToken == null) return;

      nextToken = nextToken.toLowerCase();

      if (!allLetters(nextToken)) {
        nextToken = null;
        continue;
      }

      if (stem) {
        nextToken = stemmer.stripAffixes(nextToken);
        if (!allLetters(nextToken)) {
          nextToken = null;
          continue;
        }
      }

      break;

    } while (nextToken == null);
  }
}
