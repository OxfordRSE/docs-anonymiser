## Requirements

- [Docker](https://docs.docker.com/docker-for-mac/install/)
- Bash

## Howto Name PDFs

The PDF filename must be in the following format

`Name1, Name2 Name3 - Application Dossier.pdf`

Where `Name1`, `Name2` etc are the given names of the applicant. These names will be 
redacted from the PDF, so it is essential that they are correctly spelled. Normalise any 
accents in the name (i.e. `é` becomes `e`), and do not include overly short separate 
names such as `O` in `Ó Dónaill`. The method `docs-anonymiser` uses to remove names is 
the search for a substring in each word matching one of the given names, so if `O` is 
one of the names, then any word containing `o` will also be redacted.


## To run

For PDFs that have been digitally generated, place all PDFs in a folder and run

`./docs-anonymiser.sh -d folder_path`

The character recognition will not perform well if the pages are skewed (e.g. from 
scanned PDFs) so they need to be corrected. Performing skewness correction on pages that 
do not require it, will not change them but it will increase running time (~1 min per 
page). To anonymise PDFs with skewness correction run

`./docs-anonymiser.sh -d folder_path -s`

## License

Copyright (C) 2018 Wealth Wizards, Ltd. Distributed under the terms of the [MIT License](https://opensource.org/licenses/MIT).
