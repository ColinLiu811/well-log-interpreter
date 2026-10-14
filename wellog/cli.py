"""Command line interface.

    wellog info WELL.las
    wellog interpret WELL.las --sw simandoux --plot out.png --csv out.csv
    wellog predict --train A.las B.las --input C.las --target DT --plot c.png
    wellog synth data/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .interpret import InterpretationParams, interpret
from .las_io import load_las, save_las
from .pay import Cutoffs, flag_pay, pay_zones, well_summary
from .petro import VSH_METHODS


def _cmd_info(args):
    for path in args.files:
        well = load_las(path)
        z = well.depth
        print(f"{well.name}  ({path})")
        print(f"  depth {z.min():.2f} - {z.max():.2f} {well.depth_unit}, {len(z)} samples")
        print(well.summary().to_string(float_format=lambda v: f"{v:.3f}"))
        print()


def _cmd_interpret(args):
    well = load_las(args.las)
    params = InterpretationParams(
        vsh_method=args.vsh,
        gr_clean=args.gr_clean,
        gr_shale=args.gr_shale,
        porosity_method=args.porosity,
        rho_ma=args.rho_ma,
        sw_method=args.sw,
        rw=args.rw,
        rsh=args.rsh,
        a=args.a,
        m=args.m,
        n=args.n,
    )
    cutoffs = Cutoffs(vsh_max=args.vsh_cut, phi_min=args.phi_cut, sw_max=args.sw_cut, min_thickness=args.min_thickness)

    results = flag_pay(interpret(well, params), cutoffs)
    zones = pay_zones(results, merge_gap=args.merge_gap)
    summary = well_summary(results)

    pd.set_option("display.width", 120)
    print(f"Well {well.name}")
    print("Parameters: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.to_dict().items()}))
    print(f"Net reservoir {summary['net_reservoir']:.1f} {well.depth_unit}, net pay {summary['net_pay']:.1f} {well.depth_unit}, "
          f"avg PHIE {summary['phie_avg_pay']:.3f}, avg SW {summary['sw_avg_pay']:.3f}")
    if len(zones):
        print("\nPay zones:")
        print(zones.to_string(float_format=lambda v: f"{v:.3f}"))
    else:
        print("\nNo pay zones with current cutoffs.")

    if args.csv:
        results.to_csv(args.csv)
        print(f"\nwrote {args.csv}")
    if args.zones_csv:
        zones.to_csv(args.zones_csv)
        print(f"wrote {args.zones_csv}")
    if args.las_out:
        save_las(well, args.las_out, extra=results)
        print(f"wrote {args.las_out}")
    if args.plot:
        from .plot import plot_well

        depth_range = tuple(args.depth_range) if args.depth_range else None
        plot_well(well, results, zones, depth_range=depth_range, out=args.plot)
        print(f"wrote {args.plot}")


def _cmd_predict(args):
    from .ml import LogPredictor, leave_one_well_out

    train = [load_las(p) for p in args.train]
    target = load_las(args.input) if args.input else None
    features = args.features.split(",")

    if args.cv:
        cv = leave_one_well_out(train, args.target, features)
        print("Leave-one-well-out validation:")
        print(cv.to_string(float_format=lambda v: f"{v:.3f}"))
        print()

    if args.model and Path(args.model).exists() and not args.retrain:
        predictor = LogPredictor.load(args.model)
        print(f"loaded model from {args.model}")
    else:
        fit_wells = list(train)
        if target is not None and args.self_train and target.has(args.target):
            fit_wells.append(target)
        predictor = LogPredictor(target=args.target, features=features).fit(fit_wells)
        print(f"trained {args.target} model on {len(fit_wells)} well(s)")
        if args.model:
            predictor.save(args.model)
            print(f"saved model to {args.model}")

    if target is None:
        return
    filled = predictor.fill(target)
    n_missing = target.data[args.target].isna().sum() if args.target in target.data else len(filled)
    print(f"{target.name}: predicted {args.target} for {n_missing} missing samples")

    if args.las_out:
        save_las(target, args.las_out, extra=filled)
        print(f"wrote {args.las_out}")
    if args.csv:
        filled.to_csv(args.csv)
        print(f"wrote {args.csv}")
    if args.plot:
        from .plot import plot_well

        plot_well(target, filled, out=args.plot, title=f"{target.name} - {args.target} prediction")
        print(f"wrote {args.plot}")


def _cmd_synth(args):
    from .synthetic import write_demo_dataset

    for path in write_demo_dataset(args.out_dir):
        print("wrote", path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wellog", description="Well log interpreter")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="list curves and coverage in LAS files")
    p.add_argument("files", nargs="+")
    p.set_defaults(func=_cmd_info)

    p = sub.add_parser("interpret", help="compute Vsh, porosity, Sw and pay")
    p.add_argument("las")
    g = p.add_argument_group("shale volume")
    g.add_argument("--vsh", choices=VSH_METHODS, default="linear")
    g.add_argument("--gr-clean", type=float)
    g.add_argument("--gr-shale", type=float)
    g = p.add_argument_group("porosity")
    g.add_argument("--porosity", choices=["neutron_density", "density", "sonic"], default="neutron_density")
    g.add_argument("--rho-ma", type=float, default=2.65, help="matrix density (2.65 sandstone, 2.71 limestone)")
    g = p.add_argument_group("saturation")
    g.add_argument("--sw", choices=["archie", "simandoux"], default="simandoux")
    g.add_argument("--rw", type=float, help="formation water resistivity (estimated if omitted)")
    g.add_argument("--rsh", type=float, help="shale resistivity (estimated if omitted)")
    g.add_argument("-a", type=float, default=1.0, help="tortuosity factor")
    g.add_argument("-m", type=float, default=2.0, help="cementation exponent")
    g.add_argument("-n", type=float, default=2.0, help="saturation exponent")
    g = p.add_argument_group("cutoffs")
    g.add_argument("--vsh-cut", type=float, default=0.40)
    g.add_argument("--phi-cut", type=float, default=0.08)
    g.add_argument("--sw-cut", type=float, default=0.50)
    g.add_argument("--min-thickness", type=float, default=0.5)
    g.add_argument("--merge-gap", type=float, default=1.0, help="merge pay zones separated by less than this")
    g = p.add_argument_group("output")
    g.add_argument("--csv", help="write computed curves to CSV")
    g.add_argument("--zones-csv", help="write pay zone table to CSV")
    g.add_argument("--las-out", help="write input + computed curves to LAS")
    g.add_argument("--plot", help="write the log display to an image file")
    g.add_argument("--depth-range", type=float, nargs=2, metavar=("TOP", "BASE"))
    p.set_defaults(func=_cmd_interpret)

    p = sub.add_parser("predict", help="train a model to predict a missing log")
    p.add_argument("--train", nargs="+", required=True, help="LAS files with the target curve")
    p.add_argument("--input", help="LAS file to predict the curve for")
    p.add_argument("--target", default="DT")
    p.add_argument("--features", default="GR,RHOB,NPHI,RT", help="comma-separated input curves")
    p.add_argument("--cv", action="store_true", help="report leave-one-well-out metrics")
    p.add_argument("--no-self-train", dest="self_train", action="store_false",
                   help="don't use the input well's own measured target samples for training")
    p.add_argument("--model", help="joblib path to load from / save to")
    p.add_argument("--retrain", action="store_true", help="retrain even if --model exists")
    p.add_argument("--las-out")
    p.add_argument("--csv")
    p.add_argument("--plot")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("synth", help="write the synthetic demo wells")
    p.add_argument("out_dir", nargs="?", default="data")
    p.set_defaults(func=_cmd_synth)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
