from gui import run_gui


if __name__ == '__main__':
    try:
        run_gui()
    except ValueError:
        print('ValueError')