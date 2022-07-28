import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from wordcount import run

options = PipelineOptions([
    "--runner=PortableRunner",
    "--job_endpoint=localhost:8099",
    "--environment_type=LOOPBACK"
])
with beam.Pipeline(options) as p:
    pass
    # # Read the text file[pattern] into a PCollection.
    # lines = p | 'Read' >> ReadFromText(known_args.input)

    # counts = (
        # lines
        # | 'Split' >> (beam.ParDo(WordExtractingDoFn()).with_output_types(str))
        # | 'PairWithOne' >> beam.Map(lambda x: (x, 1))
        # | 'GroupAndSum' >> beam.CombinePerKey(sum))

    # # Format the counts into a PCollection of strings.
    # def format_result(word, count):
      # return '%s: %d' % (word, count)

    # output = counts | 'Format' >> beam.MapTuple(format_result)

    # # Write the output using a "Write" transform that has side effects.
    # # pylint: disable=expression-not-assigned
    # output | 'Write' >> WriteToText(known_args.output)
