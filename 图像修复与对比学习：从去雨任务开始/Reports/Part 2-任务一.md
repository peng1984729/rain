# Part 2-任务一

```py
parser.add_argument("--model", type=str, default="baseline",
                    choices=["baseline", "unet", "DerainNet", "Mymodel"], 
                    help="Model to use")
```

![6](.\6.png)

```py
if args.model == "baseline":
	model = BaselineNet().to(device)
elif args.model == "Mymodel":
	model = BaselineNet().to(device)
elif args.model == "DerainNet":
	model = BaselineNet().to(device)
elif args.model == "unet":
    model = UNet().to(device)
```

