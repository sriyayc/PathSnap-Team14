# Data quality findings (train + student_test)

- Invalid / non-finite GPS rows dropped: 0; duplicate timestamps dropped: 0
- Trips with non-integer timestamps: ['00000002', '00000071', '00000097']
- Trips with gaps > 60 s: ['00000002', '00000006', '00000011', '00000014', '00000021', '00000024', '00000028', '00000041', '00000042', '00000043', '00000044', '00000054', '00000055', '00000069', '00000070', '00000078', '00000081', '00000084', '00000085', '00000088', '00000089', '00000090', '00000065']
- Trips where median GPS-to-route distance > 30 m (offset reference): ['00000033', '00000054']
- Trips with GPS path > 1.5x route length (route covers part of trip / stationary jitter): ['00000002', '00000003', '00000008', '00000010', '00000021', '00000033', '00000036', '00000046', '00000048', '00000054', '00000061', '00000070', '00000072', '00000075', '00000080', '00000082', '00000083', '00000084', '00000095', '00000077']
- Trips whose reference route has disconnected consecutive arcs: ['00000002', '00000019', '00000033', '00000041', '00000042', '00000069', '00000075', '00000080', '00000086', '00000089', '00000076']
- Networks with self-loops: 9; with duplicate arcs: 60
- Median share of arcs that have a reverse twin: 0.95
- Median number of weakly connected components per network: 19
