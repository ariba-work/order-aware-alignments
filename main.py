import csv
import signal
import time
from os.path import join


import click as click
from pm4py.objects.log.importer.xes.importer import apply as xes_import
from pm4py.objects.log.util.interval_lifecycle import to_interval
from pm4py.objects.petri_net.importer import importer as petri_importer
from pm4py.objects.petri_net.utils.synchronous_product import construct as construct_synchronous_product
from pm4py.util.xes_constants import DEFAULT_START_TIMESTAMP_KEY

from util.alignment_utils import construct_unfolded_net_from_alignment
from util.constants import SearchAlgorithms, PartialOrderMode, SKIP
from util.petri_net_utils import get_partial_trace_net_from_trace
from util.tools import ExecutionVariant, \
    compute_alignments_for_sync_product_net


header = ['variant', 'trace_idx', 'trace_length', 'time_taken', 'time_taken_potext', 'queued_events', 'visited_events', 'alignment_costs']


# Define a timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Trace processing timed out")

# Set the timeout duration (in seconds)
timeout_duration = 3  # Adjust as needed

# Register the timeout handler
signal.signal(signal.SIGALRM, timeout_handler)


def process_trace(trace_idx, trace, model_net, model_im, model_fm):
    try:
        # Set the alarm
        signal.alarm(timeout_duration)

        # build trace net
        net, im, fm = get_partial_trace_net_from_trace(trace, PartialOrderMode.REDUCTION, False)

        print(f'{trace_idx}::')

        # build SPN
        sync_prod, sync_im, sync_fm = construct_synchronous_product(net, im, fm, model_net, model_im, model_fm, SKIP)

        a = compute_alignments_for_sync_product_net(sync_prod, sync_im, sync_fm, ExecutionVariant(search_algorithm=SearchAlgorithms.A_STAR))

        unfolding_start_time = time.time()
        _, _, _ = construct_unfolded_net_from_alignment(a, sync_prod, sync_im)

        output = [
            0,
            trace_idx,
            len(trace),
            time.time() - unfolding_start_time + a.total_duration,
            'N/A',
            a.queued_states,
            a.visited_states,
            a.alignment_costs,
        ]

        return output

    except TimeoutError:
        print(f"Trace {trace_idx} processing timed out")
        return [
            0,
            trace_idx,
            len(trace),
            "timeout",
            "timeout",
            "timeout",
            "timeout",
            "timeout",
        ]

    finally:
        # Disable the alarm
        signal.alarm(0)


@click.command()
@click.option('--path', '-p', help='Path to the data directory.')
@click.option('--log', '-l', help='Name of the event log.')
@click.option('--model', '-m', help='Name of the process model.')
def compute_classic_alignments(path: str, log: str, model: str):

    model_net, model_im, model_fm = petri_importer.apply(join(f'.', path, model))
    event_log = xes_import(join(f'.', path, log))

    if (
            DEFAULT_START_TIMESTAMP_KEY not in event_log[0][0]
    ):
        event_log = to_interval(event_log)[:1]

    print(f'total number of traces: {len(event_log)}')

    with open(f'experiments/results/{model}.csv', mode='a', newline='') as output:
        writer = csv.writer(output)

        for trace_idx, trace in enumerate(event_log, 1):
            result = process_trace(trace_idx, trace, model_net, model_im, model_fm)
            writer.writerow(result)

        print(f'completed for model={model}, closing file')

    output.close()


@click.group()
def cli():
    pass

cli.add_command(compute_classic_alignments, "classic-alignments")

if __name__ == '__main__':
    cli()