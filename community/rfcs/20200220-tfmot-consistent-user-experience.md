# TensorFlow Model Optimization Consistent User Experience

Status        | Pending
:------------ | :------------------------------------------------------
**RFC #**     | [257](https://github.com/tensorflow/model-optimization/pull/257)
**Author(s)** | Alan Chiao (alanchiao@google.com)
**Sponsor**   | TODO
**Updated**   | 2020-02-20

## Objective
Improve the usability of the Model Optimization Toolkit, by making the user
experience consistent across techniques whenever it is sensible. “User” refers
to the end user of the technique. Other types of users are out of scope.

## Design Proposal

At a high-level, achieving this requires the following:

1. Similar documentation structure across techniques, so that users can easily determine what
techniques they want to try and then try them.

2. Clear expectations on the degree of functionality and support that users should expect
from the technique and its OWNERS. We achieve this through defining what ownership
means. The [technique contribution guide](https://github.com/tensorflow/model-optimization/blob/master/CONTRIBUTING_TECHNIQUE.md)
helps the technique contributor achieve this, though the user should never have to read that guide
(TODO: link to ownership RFC).

3. Similar public APIs when sensible. We achieve this through the technique
contribution RFC process (TODO: link to RFC process).
   * An API built on tf.keras will be different from an API built on core TensorFlow.
  However, Keras APIs for techniques with shared properties should be as similar
  as possible in API structure and style, without compromising on the needs of the
  users of the technique itself.

The rest of this document focuses on 1.

### Documentation Structure Proposal

#### User Needs

1. They want to determine whether or not they want to use the technique and
   tooling in the first place.

2. They want to see a demonstration of the benefits and an end-to-end path to
   get there.

3. They want to find the specific API classes and functions they need for their
   varying use cases (e.g. deployment versus research,
   Sequential/Functional/Subclassed, model.fit/custom training loop).

For TFMOT owners, it's important for documentation to be maintainable. Reducing
duplicate information helps here.

User need #2 can be in conflict with need #3. In an end-to-end example from #2, it
can be hard to find the exact class or function given where the user is at (e.g.
the user has a trained model already. They just need to deploy it.). At some
point, the boilerplate code doesn't provide useful context and adds noise. This
can be even harder with multiple end-to-end examples.

#### Design

Each technique will have three different documentation pages that each focuses on one of the three
user needs.

1. Overview page: users determine whether they want to try the technique and tool in the first place.  Given its purpose:
  * Once users decide to try the technique, the page guides them documentation
    for using the tool.
  * The page should be as non-technical as possible. Most of the “how a
    technique works” is not relevant to the decision making process.

  The "Tips" section in the current pruning overview page is moved elsewhere. Tips do not help
  much with user need #1, even if they signal ease of use.

2. End-to-end example in Colab: demonstrate the benefits and end-to-end-path to
   get there. A “golden” path.
   * The demonstrated path is the most critical path, with respect to TFMOT’s
     focus on easily deploying more efficient models.
   * The benefits are primarily in page #1 and not here, though some will be
     mentioned because the Colab demonstrates them.

   From the existing pruning example, a lot of the
   content available in the overview page is de-duplicated.
   Instead of demonstrating two paths at once (which delays the time to
   demonstrate the final benefits), the example only demonstrates one path.

3. Comprehensive guide in Colab: users find the specific API classes and
   functions they need for their varying use cases.

   * Conciseness is key. This reiterates the disadvantage of an end-to-end
     example where it can be hard to find the exact few lines of code amongst
     boilerplate and lots of text.

   * Major headers for major use cases, which can be divided into smaller use
     cases. For an example, users of quantization may want to either deploy with
     quantization or research quantization. There could be one header for each
     and then further divisions based on deployment and research needs.

   *  There are navigational and non-navigational tips
      with regards to finding and using the APIs they need. For instance, "Pruning all
      layers" would provide a tip to navigate to "Prune subset of layers" if
      model accuracy is not sufficient. The navigational ones should
      be placed on top of the non-navigational ones.

   * This is a Colab, since there is infrastructure to detect when Colabs fail
     to run through, which is more maintainable.

      * As a Colab, there has to be boilerplate for standard Keras steps. While
        each subsection provides some boilerplate for context, there is also
        a dedicated boilerplate code section to run once per Colab session.

General tradeoffs that apply to all:

   * Navigation-first over minimal to no navigation. For instance, the overview page can
     either start with its core content or links to the other pages that serve
     other purposes.
     * Con: it delays people who know where they are from getting to the main
       content.
     * Pro: it speeds up things for people who aren’t at the right place yet.
       Links to the different pages and subsections can be shared in several
       ways and people can land on the page that doesn’t address their
       needs.

     Conclusion: navigation-first as a priority.

   * Is the end-to-end example worth maintaining in addition to the
     comprehensive guide? These are benefits of the end-to-end example:

       * It provides a cohesive story without mixing and matching subsections.
       * It can allow for a more resource-intensive example (e.g. MNIST or
         larger) that would otherwise detract from users finding the APIs they
         need in the comprehensive guide.
       * It is an appropriate place to demonstrate accuracy results and
         backend-dependent latency benefits.

     Conclusion: the end-to-end example is worth it.









