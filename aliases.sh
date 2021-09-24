#!/bin/zsh

# Перейти на рабочий стол
alias cd_desktop='cd ~; cd ./Desktop; pwd'

alias ssh_amazon='ssh -i ~/.ssh/amazon_server_ssh_keys.cer ec2-user@ec2-18-118-100-1.us-east-2.compute.amazonaws.com'

get_from_amazon() {
	scp -i ~/.ssh/amazon_server_ssh_keys.cer -r ec2-user@ec2-18-118-100-1.us-east-2.compute.amazonaws.com:$1 $2
}

put_to_amazon() {
	scp -i ~/.ssh/amazon_server_ssh_keys.cer -r $1 ec2-user@ec2-18-118-100-1.us-east-2.compute.amazonaws.com:$2
}


# Открывает файл в яндекс браузере
alias yandex_open='/Applications/Yandex.app/Contents/MacOS/Yandex'


# Открывает ноутбук в Jetbrains Dataspell
alias dataspell='/Applications/JetBrains\ DS\ 2021.3.app/Contents/MacOS/dataspell'


# Создаёт HTML репорт по датафрейму (докер версия)
# params вида: --file='<относительный путь до файла>' --sep='<разделитель, по дефолту запятая>' --target='<столбец с таргетом>'

function fast_eda {
	params=$*
	docker run --rm -it -e a=$params -v $(PWD):/tmp joitandr/analyze_csv /bin/bash -c 'cd tmp; python3 ../app/analyze_csv.py $a'
}


# стартует докер из командной строки
alias start_docker='open /Applications/Docker.app/'

# открывает jupyter в докере (custom_ds_image)
alias custom_jup_docker='cd /Users/antonandreytsev/Desktop; echo "http://127.0.0.1:8989/?token=joitandr1410" ; docker run --rm -it -v $PWD:/opt/notebooks -p 8989:8989 custom_ds_image'

# открывает jupyter в докере (msu_prac_trees)
alias jup_trees='cd /Users/antonandreytsev/Desktop/msu_prac_trees; echo "http://127.0.0.1:8989/?token=joitandr1410" ; docker run --rm -it -v $PWD:/prac_folder -p 8989:8989 msu_prac_trees'


# достаёт строки с определёнными номера из stdout 
# применение: <запрос> | eval $(grep_rows <номера строк через пробел>)
# пример: ps aux | grep '{print $1, $2}' | eval $(grep_rows 1 3 24)

grep_rows() {
  declare rows_array=($*)
  grep_string="^\("
  for elem in ${rows_array[@]}
  do
    grep_string="$grep_string$elem\|"
  done
  grep_string="${grep_string:0:-1}\):"
  returned_string="grep -n '.*' | grep '$grep_string'"
  echo $returned_string
}


export DOCKER_FORMAT="\nID\t{{.ID}}\nIMAGE\t{{.Image}}\nCOMMAND\t{{.Command}}\nCREATED\t{{.RunningFor}}\nSTATUS\t{{.Status}}\nPORTS\t{{.Ports}}\nNAMES\t{{.Names}}\n"

# Создаёт случайный текстовый документ
# пример использования: 
# 1) text
# 2) text py (открывает в pycharm)
# 3) text sh (открывает в pycharm)
# alias text='bash ~/.make_text_file.sh'



# Алиасы для gshell команд:
alias cd_g='gshell cd'
alias pwd_g='gshell pwd'
alias ls_g='gshell ls'
alias ll_g='gshell ll'
alias rm_g='gshell rm'
alias upload_g='gshell upload' 
alias download_g='gshell download'



alias speedtest='curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python -'

# Распаковка архивов
# example: extract file
extract () {
 if [ -f $1 ] ; then
 case $1 in
 *.tar.bz2)   tar xjf $1        ;;
 *.tar.gz)    tar xzf $1     ;;
 *.bz2)       bunzip2 $1       ;;
 *.rar)       unrar x $1     ;;
 *.gz)        gunzip $1     ;;
 *.tar)       tar xf $1        ;;
 *.tbz2)      tar xjf $1      ;;
 *.tbz)       tar -xjvf $1    ;;
 *.tgz)       tar xzf $1       ;;
 *.zip)       unzip $1     ;;
 *.Z)         uncompress $1  ;;
 *.7z)        7z x $1    ;;
 *)           echo "I don't know how to extract '$1'..." ;;
 esac
 else
 echo "'$1' is not a valid file"
 fi
}


# Вольфрам вычисления в консоли https://support.wolfram.com/46070

alias wf="wolframscript --code $(echo $* | tr -d ' ')"

# Вольфрам вычисление из файла .wolfram_file.wl

alias wff='echo; wolframscript -code -file ~/Desktop/.wolfram_file.wl | sed 's/Null//g''


# ssh доступ к хадупу ШАДа

alias ssh_shad='ssh aandreytsev@hadoop2.yandex.ru'


# Перевести .ipynb в .html

alias jup2html='jupyter nbconvert --to=html'


# Воспроизводит Федота стрельца в VLC
alias fedot_strelez='open /Users/antonandreytsev/Desktop/.Leonid_ilatov_-_Skaz_pro_edota-strelca_70110477.mp3 -a vlc'


# Открыть файл в VLC
alias vlc='/Applications/VLC.app/Contents/MacOS/VLC'


# Переводит консольный вывод в .png картинку
alias output2png='convert label:@- tmp_tmp_tmp.png; open ./tmp_tmp_tmp.png'


# Кладёт вывод консоли в termbin.com
alias tb='nc termbin.com 9999 | pbcopy'

# Открывает файл в PyCharm
alias pycharm='open -a /Applications/PyCharm.app/Contents/MacOS/pycharm'


# убиваем все docker-контейнеры
alias docker_rm_containers='docker rm -f $(docker ps -a -q)'

# таймер
function timer {
	sleep $1
	afplay -v 4 /Users/antonandreytsev/Desktop/.alarm_sound.mp3
}




# Создаёт случайный текстовый документ =======================================================================

function text {

  declare -a arr=( $( echo "$@" | cut -d' ' -f1- ) )
#  echo "len(arr): ${#arr[@]}"
  len_arr=${#arr[@]}
  if [[ $len_arr -eq 0 ]]
  then
    cd /Users/antonandreytsev/Desktop/
    file_name="/Users/antonandreytsev/Desktop/tmp_$RANDOM.txt"
    touch $file_name
    echo $file_name
    open -e $file_name
  fi

  if [[ $len_arr -eq 1 ]]
  then

    # Если аргумент это py или sh:
    if [[ "${arr[1]}" =~ "^py$|^sh$" ]]
    then
#      echo "Аргумент это py или sh"
      file_type=${arr[1]}
      rando=$RANDOM
      tmp_name=tmp_$rando
      echo $tmp_name.$file_type
      cd /Users/antonandreytsev/Desktop

      case "$file_type" in
       "py")
            touch $tmp_name.py; open -a /Applications/PyCharm.app/Contents/MacOS/pycharm $tmp_name.py
            echo python3 $tmp_name.py | pbcopy
            ;;
       "sh")
            echo "#!/bin/zsh" > $tmp_name.sh; open -a /Applications/PyCharm.app/Contents/MacOS/pycharm $tmp_name.sh
            echo zsh $tmp_name.sh | pbcopy
            ;;
      esac
    else
      # Если в аргументе есть путь до файла ------------------------------------------------------------------------
      if [[ "${arr[1]}" =~ '/' ]]
  #    echo "Есть путь до файла"
      then
  #      echo "в названии есть /"
        if [[ "${arr[1]}" =~ ".py$|.txt$|.sh$" ]]
        then
          if [[ "${arr[1]}" =~ ".txt$" ]]
          then
            touch ${arr[1]}
            echo ${arr[1]}
            open -e ${arr[1]}
          fi

          if [[ "${arr[1]}" =~ ".py$" ]]
          then
            touch ${arr[1]}
            echo ${arr[1]}
            open -a /Applications/PyCharm.app/Contents/MacOS/pycharm ${arr[1]}; echo python3 ${arr[1]} | pbcopy
          fi

          if [[ "${arr[1]}" =~ ".sh$" ]]
          then
            touch ${arr[1]}
            echo ${arr[1]}
            echo "#!/bin/zsh" > ${arr[1]}
            open -a /Applications/PyCharm.app/Contents/MacOS/pycharm ${arr[1]}; echo zsh ${arr[1]} | pbcopy
          fi
        else
          touch "${arr[1]}.txt"
          echo "${arr[1]}.txt"
          open -e "${arr[1]}.txt"
        fi
      # Если нет пути до файла ------------------------------------------------------------------------------------------
      else
  #      echo "Нет пути до файла"
        #cd /Users/antonandreytsev/Desktop/
        if [[ "${arr[1]}" =~ ".py$|.txt$|.sh$" ]]
        then

          if [[ "${arr[1]}" =~ ".txt$" ]]
          then
            touch "${arr[1]}"
            echo "${arr[1]}"
            open -e "${arr[1]}"
          fi

          if [[ "${arr[1]}" =~ ".py$" ]]
          then
            touch ${arr[1]}
            echo ${arr[1]}
            open -a /Applications/PyCharm.app/Contents/MacOS/pycharm ${arr[1]}
            echo python3 ${arr[1]} | pbcopy
          fi

          if [[ "${arr[1]}" =~ ".sh$" ]]
          then
            touch ${arr[1]}
            echo ${arr[1]}
            echo "#!/bin/zsh" > ${arr[1]}
            open -a /Applications/PyCharm.app/Contents/MacOS/pycharm ${arr[1]}
            echo zsh ${arr[1]} | pbcopy
          fi

        else
          touch "${arr[1]}.txt"
          echo "${arr[1]}.txt"
          open -e "${arr[1]}.txt"
        fi
      fi
    fi
    # ------------------------------------------------------------------------------------------------------------
  fi

  if [[ $len_arr -eq 2 ]]
  then
    file_type=${arr[1]}
    rando=$RANDOM
    tmp_name=tmp_$rando
    echo $tmp_name
    cd /Users/antonandreytsev/Desktop

    case "$file_type" in
     "py")
          touch $tmp_name.py; open -a /Applications/PyCharm.app/Contents/MacOS/pycharm $tmp_name.py
          echo python3 $tmp_name.py | pbcopy
          ;;
     "sh")
          echo "#!/bin/zsh" > $tmp_name.sh; open -a /Applications/PyCharm.app/Contents/MacOS/pycharm $tmp_name.sh
          echo zsh $tmp_name.sh | pbcopy
          ;;
     *)
          touch $tmp_name.txt
          open -e $tmp_name.txt
          ;;
    esac
  fi

}

# tests -----------------------------------
#text /Users/antonandreytsev/texmo
#text /Users/antonandreytsev/texmo.txt
#text /Users/antonandreytsev/texmo.sh
#text /Users/antonandreytsev/texmo.py
#text some_text_file1
#text some_text_file1.txt
#text some_text_file1.py
#text some_text_file1.sh
#text
#text py
#text sh
#----------------------------------------

#====================================================================================================================
