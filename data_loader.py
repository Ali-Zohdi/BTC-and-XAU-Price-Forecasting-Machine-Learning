import pandas as pd
import numpy as np

def data_loader(data_path, weather_path, X_marking, weeks_lookback, stacked, batch_size, slice):
    """
    Parameters
    ----------
    data_path : str
        the path to the generation or load dataset formated in csv

    weather_path : str
        the path to the weather dataset formated in csv
    
    X_marking : bool
        if True -> the x_mark dataset will be provided
        if False -> the 'mark' of the X data is included in the X and no x_mark is provided
    
    weeks_lookback : int
        number of weeks that are included on the X

    stacked : bool
        if True -> return dataset as (B, N, L, C)
        if False -> return dataset as (B, N*L, C)

    batch_size : int
        create data loader with batch size

    Returns
    -------
    (train_loader, validation_loader, test_loader)
    -> Each loader contains 4 Tensors: X, X_mark, Y, Y_mark

    """

    main_df = pd.read_csv(data_path)
    weather_df = pd.read_csv(weather_path)

    household_count = len(main_df['household'].unique())

    X = []
    X_mark = []
    y = []
    y_mark = []

    i = 1
    while i < len(weather_df['date_index'].unique()) - weeks_lookback + 1:
    # print(i, i+1, i+2, i+3," -> ", i+4)

        for household_index, name_ in enumerate(main_df[main_df['date_index'] == i]['household'].unique()):

            sliced_gen = main_df[main_df['household'] == f'{name_}']
            # min_date_index = min(sliced_gen['date_index'])

            all_weeks = []
            all_weeks_mark = []

            # 'weeks_lookback' WEEKS BEFORE
            for week_index in range(weeks_lookback):
                one_week = []
                one_week_mark = []
                sliced_weath = weather_df[weather_df['date_index'] == i + week_index]

                one_week.append(list(sliced_gen[sliced_gen['date_index'] == i + week_index].iloc[0, -168:]))     # Generation of the Week
                one_week.append(list(sliced_weath.loc[:, 'DE_temperature']))                                     # Weather of the Week
                one_week.append(list(sliced_weath.loc[:, 'DE_radiation_direct_horizontal']))                     # Radiation Direction Horizon of the Week
                one_week.append(list(sliced_weath.loc[:, 'DE_radiation_diffuse_horizontal']))                    # Radiation Diffuse Horizon of the Week

                if X_marking:
                    one_week_mark.append([household_index/household_count for r in range(168)])                           # X_mark: pv index
                    one_week_mark.append(list(sliced_weath.loc[:, 'Month']))                                              # X_mark: month
                    one_week_mark.append(list(sliced_weath.loc[:, 'WeekDay']))                                            # X_mark: day of the week
                    one_week_mark.append(list(sliced_weath.loc[:, 'HourofDay']))                                          # X_mark: hour of the day
                else:
                    one_week.append([household_index/household_count for r in range(168)])                           # X_mark: pv index
                    one_week.append(list(sliced_weath.loc[:, 'Month']))                                              # X_mark: month
                    one_week.append(list(sliced_weath.loc[:, 'WeekDay']))                                            # X_mark: day of the week
                    one_week.append(list(sliced_weath.loc[:, 'HourofDay']))                                          # X_mark: hour of the day

                all_weeks.append(list(map(list, zip(*one_week))))
                all_weeks_mark.append(list(map(list, zip(*one_week_mark))))

            X.append(all_weeks)
            X_mark.append(all_weeks_mark)

            # 1 WEEK PREDICTION
            y.append(list(sliced_gen[sliced_gen['date_index'] == i + weeks_lookback].iloc[0, -168:]))

            sliced_weath = weather_df[weather_df['date_index'] == i + weeks_lookback]
            mark_list = []
            mark_list.append([household_index/household_count for r in range(168)])                           # y_mark: pv index
            mark_list.append(list(sliced_weath.loc[:, 'Month']))                                              # y_mark: month
            mark_list.append(list(sliced_weath.loc[:, 'WeekDay']))                                            # y_mark: day of the week
            mark_list.append(list(sliced_weath.loc[:, 'HourofDay']))                                          # y_mark: hour of the day
            y_mark.append(list(map(list, zip(*mark_list))))

        if i + weeks_lookback == slice[0]: # Train
            X_train = X
            X_mark_train = X_mark
            y_train = y
            y_mark_train = y_mark

            X = []
            X_mark = []
            y = []
            y_mark = []

        elif i + weeks_lookback == slice[1]: # Validation
            X_valid = X
            X_mark_valid = X_mark
            y_valid = y
            y_mark_valid = y_mark

            X = []
            X_mark = []
            y = []
            y_mark = []

        elif i + weeks_lookback == slice[2]: # Test
            X_test = X
            X_mark_test = X_mark
            y_test = y
            y_mark_test = y_mark

            X = []
            X_mark = []
            y = []
            y_mark = []

        i += 1

    X_Train = np.array(X_train)
    X_mark_Train = np.array(X_mark_train)
    y_Train = np.array(y_train)
    y_mark_Train = np.array(y_mark_train)

    X_Valid = np.array(X_valid)
    X_mark_Valid = np.array(X_mark_valid)
    y_Valid = np.array(y_valid)
    y_mark_Valid = np.array(y_mark_valid)

    X_Test = np.array(X_test)
    X_mark_Test = np.array(X_mark_test)
    y_Test = np.array(y_test)
    y_mark_Test = np.array(y_mark_test)

    if not stacked:

        B_tr, N_tr, L_tr, C_tr = X_Train.shape
        B_va, N_va, L_va, C_va = X_Valid.shape
        B_te, N_te, L_te, C_te = X_Test.shape    

        X_Train = X_Train.reshape(B_tr, N_tr * L_tr, C_tr)
        X_Valid = X_Valid.reshape(B_va, N_va * L_va, C_va)
        X_Test = X_Test.reshape(B_te, N_te * L_te, C_te) 

    print(f"_____TRAIN__________________\nX     : {X_Train.shape}\nX_mark: {X_mark_Train.shape}\ny     : {y_Train.shape}\ny_mark: {y_mark_Train.shape}\n")
    print(f"_____VALIDATION_____________\nX     : {X_Valid.shape}\nX_mark: {X_mark_Valid.shape}\ny     : {y_Valid.shape}\ny_mark: {y_mark_Valid.shape}\n")
    print(f"_____TEST___________________\nX     : {X_Test.shape}\nX_mark: {X_mark_Test.shape}\ny     : {y_Test.shape}\ny_mark: {y_mark_Test.shape}\n")               

    X_train = torch.from_numpy(X_Train).to(torch.float32)
    X_mark_train = torch.from_numpy(X_Train).to(torch.float32)
    Y_train = torch.from_numpy(y_Train).to(torch.float32)
    Y_mark_train = torch.from_numpy(y_Train).to(torch.float32)

    X_test = torch.from_numpy(X_Test).to(torch.float32)
    X_mark_test = torch.from_numpy(X_Test).to(torch.float32)
    Y_test = torch.from_numpy(y_Test).to(torch.float32)
    Y_mark_test = torch.from_numpy(y_Test).to(torch.float32)

    X_valid = torch.from_numpy(X_Valid).to(torch.float32)
    X_mark_valid = torch.from_numpy(X_Valid).to(torch.float32)
    Y_valid = torch.from_numpy(y_Valid).to(torch.float32)
    Y_mark_valid = torch.from_numpy(y_Valid).to(torch.float32)

    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(X_train, X_mark_train, Y_train, Y_mark_train)
    test = torch.utils.data.TensorDataset(X_test, X_mark_test, Y_test, Y_mark_test)
    valid = torch.utils.data.TensorDataset(X_valid, X_mark_valid, Y_valid, Y_mark_valid)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size = batch_size, shuffle = False)   

    return train_loader, valid_loader, test_loader